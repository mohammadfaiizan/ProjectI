"""
Dynamic Programming - Game Theory Patterns
This module implements various game theory problems with DP including Stone Game variants,
Nim Game variations, optimal strategy games, and Grundy number computations.
"""

from typing import List, Dict, Tuple, Optional, Set
import time
from functools import lru_cache

# ==================== STONE GAME VARIANTS ====================

class StoneGameVariants:
    """
    Stone Game Problem Variants
    
    Various versions of the classic stone game where two players
    alternately pick stones to maximize their own score.
    """
    
    def stone_game_i(self, piles: List[int]) -> bool:
        """
        Stone Game I - Pick from ends only
        
        LeetCode 877 - Stone Game
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        
        Args:
            piles: Array of stone piles
        
        Returns:
            True if Alex (first player) wins
        """
        n = len(piles)
        
        # Mathematical insight: Alex always wins when n is even
        # because she can choose to take all even indices or all odd indices
        if n % 2 == 0:
            return True
        
        # General DP solution for odd n or educational purpose
        # dp[i][j] = max stones Alex can get more than Lee for piles[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Base case: single pile
        for i in range(n):
            dp[i][i] = piles[i]
        
        # Fill DP table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Alex picks from left: gets piles[i], Lee plays optimally on [i+1, j]
                pick_left = piles[i] - dp[i + 1][j]
                
                # Alex picks from right: gets piles[j], Lee plays optimally on [i, j-1]
                pick_right = piles[j] - dp[i][j - 1]
                
                dp[i][j] = max(pick_left, pick_right)
        
        return dp[0][n - 1] > 0
    
    def stone_game_ii(self, piles: List[int]) -> int:
        """
        Stone Game II - Variable number of piles
        
        LeetCode 1140 - Stone Game II
        
        Players can take 1 to 2*M piles from the beginning.
        M starts at 1 and gets updated to max(M, X) where X is piles taken.
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            piles: Array of stone piles
        
        Returns:
            Maximum stones Alex can collect
        """
        n = len(piles)
        
        # Suffix sums for quick calculation
        suffix_sum = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + piles[i]
        
        # Memoization
        memo = {}
        
        def dp(i: int, m: int) -> int:
            """
            Returns max stones current player can get from piles[i:] with parameter m
            """
            if i >= n:
                return 0
            
            if (i, m) in memo:
                return memo[(i, m)]
            
            # If remaining piles <= 2*m, take all
            if i + 2 * m >= n:
                memo[(i, m)] = suffix_sum[i]
                return suffix_sum[i]
            
            # Try taking x piles (1 <= x <= 2*m)
            min_opponent_gain = float('inf')
            
            for x in range(1, 2 * m + 1):
                if i + x <= n:
                    # Opponent plays optimally on remaining piles
                    opponent_gain = dp(i + x, max(m, x))
                    min_opponent_gain = min(min_opponent_gain, opponent_gain)
            
            # Current player gets: total remaining - opponent's optimal gain
            result = suffix_sum[i] - min_opponent_gain
            memo[(i, m)] = result
            return result
        
        return dp(0, 1)
    
    def stone_game_iii(self, stone_values: List[int]) -> str:
        """
        Stone Game III - Take 1, 2, or 3 stones from beginning
        
        LeetCode 1406 - Stone Game III
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            stone_values: Array of stone values (can be negative)
        
        Returns:
            "Alice", "Bob", or "Tie"
        """
        n = len(stone_values)
        
        # dp[i] = max score difference (current_player - opponent) from position i
        dp = [0] * (n + 3)  # Extra space to avoid boundary checks
        
        # Work backwards
        for i in range(n - 1, -1, -1):
            # Try taking 1, 2, or 3 stones
            take_one = stone_values[i] - dp[i + 1]
            take_two = sum(stone_values[i:i + 2]) - dp[i + 2] if i + 1 < n else stone_values[i]
            take_three = sum(stone_values[i:i + 3]) - dp[i + 3] if i + 2 < n else sum(stone_values[i:])
            
            dp[i] = max(take_one, take_two, take_three)
        
        if dp[0] > 0:
            return "Alice"
        elif dp[0] < 0:
            return "Bob"
        else:
            return "Tie"
    
    def stone_game_iv(self, n: int) -> bool:
        """
        Stone Game IV - Remove perfect square number of stones
        
        LeetCode 1510 - Stone Game IV
        
        Time Complexity: O(n√n)
        Space Complexity: O(n)
        
        Args:
            n: Number of stones initially
        
        Returns:
            True if Alice (first player) wins with optimal play
        """
        # dp[i] = True if current player wins with i stones
        dp = [False] * (n + 1)
        
        for i in range(1, n + 1):
            # Try removing each perfect square <= i
            square = 1
            while square * square <= i:
                # If opponent loses after we remove square*square stones, we win
                if not dp[i - square * square]:
                    dp[i] = True
                    break
                square += 1
        
        return dp[n]
    
    def stone_game_v(self, stone_values: List[int]) -> int:
        """
        Stone Game V - Divide into two non-empty rows
        
        LeetCode 1563 - Stone Game V
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            stone_values: Array of stone values
        
        Returns:
            Maximum score Alice can achieve
        """
        n = len(stone_values)
        
        # Prefix sums for quick range sum calculation
        prefix_sum = [0]
        for val in stone_values:
            prefix_sum.append(prefix_sum[-1] + val)
        
        def range_sum(i: int, j: int) -> int:
            return prefix_sum[j + 1] - prefix_sum[i]
        
        # Memoization
        memo = {}
        
        def dp(i: int, j: int) -> int:
            """
            Maximum score achievable from stones[i:j+1]
            """
            if i == j:
                return 0  # Can't divide single stone
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            max_score = 0
            
            # Try all possible division points
            for k in range(i, j):
                left_sum = range_sum(i, k)
                right_sum = range_sum(k + 1, j)
                
                if left_sum < right_sum:
                    # Alice takes left part and continues game on left
                    score = left_sum + dp(i, k)
                elif left_sum > right_sum:
                    # Alice takes right part and continues game on right
                    score = right_sum + dp(k + 1, j)
                else:
                    # Equal sums, Alice chooses the better option
                    score = left_sum + max(dp(i, k), dp(k + 1, j))
                
                max_score = max(max_score, score)
            
            memo[(i, j)] = max_score
            return max_score
        
        return dp(0, n - 1)

# ==================== NIM GAME VARIANTS ====================

class NimGameVariants:
    """
    Nim Game and its variants
    
    Classical game theory problems involving taking objects
    from piles with various rules and constraints.
    """
    
    def nim_game_basic(self, n: int) -> bool:
        """
        Basic Nim Game - Remove 1, 2, or 3 stones
        
        LeetCode 292 - Nim Game
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        
        Args:
            n: Number of stones
        
        Returns:
            True if first player can win
        """
        # Mathematical insight: first player loses if n % 4 == 0
        return n % 4 != 0
    
    def nim_game_general(self, piles: List[int]) -> bool:
        """
        General Nim Game with multiple piles
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            piles: List of pile sizes
        
        Returns:
            True if first player can win
        """
        # Sprague-Grundy theorem: XOR all pile sizes
        xor_sum = 0
        for pile in piles:
            xor_sum ^= pile
        
        return xor_sum != 0
    
    def nim_game_misere(self, piles: List[int]) -> bool:
        """
        Misère Nim - Player who takes last stone loses
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            piles: List of pile sizes
        
        Returns:
            True if first player can win in misère version
        """
        xor_sum = 0
        piles_gt_1 = 0
        
        for pile in piles:
            xor_sum ^= pile
            if pile > 1:
                piles_gt_1 += 1
        
        # If all piles have size 1, winner is determined by parity
        if piles_gt_1 == 0:
            return len(piles) % 2 == 0
        
        # Otherwise, use normal nim strategy
        return xor_sum != 0
    
    def nim_game_kayles(self, pins: int) -> bool:
        """
        Kayles - Bowling pin game variant of Nim
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            pins: Number of pins in a row
        
        Returns:
            True if first player can win
        """
        if pins <= 0:
            return False
        
        # DP to compute Grundy numbers
        grundy = [0] * (pins + 1)
        
        for i in range(1, pins + 1):
            moves = set()
            
            # Remove one pin: split into two parts
            for j in range(i):
                left = j
                right = i - j - 1
                moves.add(grundy[left] ^ grundy[right])
            
            # Remove two adjacent pins: split into two parts
            for j in range(i - 1):
                left = j
                right = i - j - 2
                moves.add(grundy[left] ^ grundy[right])
            
            # Find mex (minimum excludant)
            mex = 0
            while mex in moves:
                mex += 1
            
            grundy[i] = mex
        
        return grundy[pins] != 0
    
    def nim_game_with_constraints(self, piles: List[int], 
                                max_take: List[int]) -> bool:
        """
        Nim with different maximum take constraints per pile
        
        Time Complexity: O(sum(piles))
        Space Complexity: O(max(piles))
        
        Args:
            piles: List of pile sizes
            max_take: Maximum stones that can be taken from each pile
        
        Returns:
            True if first player can win
        """
        def grundy_number(pile_size: int, max_take_val: int) -> int:
            """Compute Grundy number for a single pile"""
            if pile_size == 0:
                return 0
            
            memo = {}
            
            def compute_grundy(size: int) -> int:
                if size == 0:
                    return 0
                
                if size in memo:
                    return memo[size]
                
                moves = set()
                for take in range(1, min(size, max_take_val) + 1):
                    moves.add(compute_grundy(size - take))
                
                # Find mex
                mex = 0
                while mex in moves:
                    mex += 1
                
                memo[size] = mex
                return mex
            
            return compute_grundy(pile_size)
        
        # XOR all Grundy numbers
        xor_sum = 0
        for pile, max_take_val in zip(piles, max_take):
            xor_sum ^= grundy_number(pile, max_take_val)
        
        return xor_sum != 0

# ==================== GENERAL GAME THEORY PROBLEMS ====================

class GeneralGameTheory:
    """
    General Game Theory Problems
    
    Various strategic games that can be solved using DP
    and game theory principles.
    """
    
    def can_i_win(self, max_choosable_integer: int, desired_total: int) -> bool:
        """
        Can I Win game
        
        LeetCode 464 - Can I Win
        
        Time Complexity: O(2^n * n)
        Space Complexity: O(2^n)
        
        Args:
            max_choosable_integer: Maximum integer that can be chosen
            desired_total: Target total to reach
        
        Returns:
            True if first player can force a win
        """
        # Quick checks
        total_sum = max_choosable_integer * (max_choosable_integer + 1) // 2
        if total_sum < desired_total:
            return False
        
        if max_choosable_integer >= desired_total:
            return True
        
        # Use bitmask to represent used numbers
        memo = {}
        
        def can_win(used_mask: int, current_total: int) -> bool:
            if current_total >= desired_total:
                return False  # Previous player already won
            
            if used_mask in memo:
                return memo[used_mask]
            
            # Try each unused number
            for i in range(max_choosable_integer):
                if not (used_mask & (1 << i)):  # Number i+1 not used
                    new_total = current_total + (i + 1)
                    new_mask = used_mask | (1 << i)
                    
                    # If this move wins immediately or opponent can't win after this move
                    if new_total >= desired_total or not can_win(new_mask, new_total):
                        memo[used_mask] = True
                        return True
            
            memo[used_mask] = False
            return False
        
        return can_win(0, 0)
    
    def predict_the_winner_general(self, nums: List[int]) -> bool:
        """
        Generalized Predict the Winner
        
        LeetCode 486 - Predict the Winner
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        n = len(nums)
        
        # dp[i][j] = max advantage first player can achieve on nums[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Base case: single element
        for i in range(n):
            dp[i][i] = nums[i]
        
        # Fill DP table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Pick from left
                pick_left = nums[i] - dp[i + 1][j]
                
                # Pick from right
                pick_right = nums[j] - dp[i][j - 1]
                
                dp[i][j] = max(pick_left, pick_right)
        
        return dp[0][n - 1] >= 0
    
    def flip_game_ii(self, current_state: str) -> bool:
        """
        Flip Game II - Optimal strategy game
        
        LeetCode 294 - Flip Game II
        
        Time Complexity: O(n!! ) where n!! is double factorial
        Space Complexity: O(n)
        
        Args:
            current_state: String of '+' and '-'
        
        Returns:
            True if current player can guarantee a win
        """
        memo = {}
        
        def can_win(state: str) -> bool:
            if state in memo:
                return memo[state]
            
            # Try all possible moves
            for i in range(len(state) - 1):
                if state[i] == '+' and state[i + 1] == '+':
                    # Make the move
                    new_state = state[:i] + '--' + state[i + 2:]
                    
                    # If opponent can't win from new state, current player wins
                    if not can_win(new_state):
                        memo[state] = True
                        return True
            
            memo[state] = False
            return False
        
        return can_win(current_state)
    
    def cat_and_mouse_game(self, graph: List[List[int]]) -> int:
        """
        Cat and Mouse Game
        
        LeetCode 913 - Cat and Mouse
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            graph: Adjacency list representation of the graph
        
        Returns:
            1 if mouse wins, 2 if cat wins, 0 if draw
        """
        n = len(graph)
        
        # dp[mouse][cat][turn] = result when mouse at mouse, cat at cat, turn's move
        # turn: 1 = mouse, 2 = cat
        dp = [[[0] * 3 for _ in range(n)] for _ in range(n)]
        degree = [[[0] * 3 for _ in range(n)] for _ in range(n)]
        
        # Calculate degrees (number of possible moves)
        for m in range(n):
            for c in range(n):
                degree[m][c][1] = len(graph[m])
                degree[m][c][2] = len(graph[c])
                for node in graph[c]:
                    if node == 0:  # Cat can't go to hole
                        degree[m][c][2] -= 1
        
        # Initialize winning/losing positions
        from collections import deque
        queue = deque()
        
        for cat in range(n):
            for turn in range(1, 3):
                # Mouse wins when it reaches hole
                dp[0][cat][turn] = 1
                queue.append((0, cat, turn))
                
                if cat > 0:  # Cat can't be at hole
                    # Cat wins when it catches mouse
                    dp[cat][cat][turn] = 2
                    queue.append((cat, cat, turn))
        
        # Backward propagation
        while queue:
            mouse, cat, turn = queue.popleft()
            
            if turn == 1:  # Mouse's turn
                for prev_mouse in graph[mouse]:
                    if dp[prev_mouse][cat][2] == 0:
                        if dp[mouse][cat][1] == 1:  # Mouse wins
                            dp[prev_mouse][cat][2] = 1
                            queue.append((prev_mouse, cat, 2))
                        else:
                            degree[prev_mouse][cat][2] -= 1
                            if degree[prev_mouse][cat][2] == 0:
                                dp[prev_mouse][cat][2] = 2
                                queue.append((prev_mouse, cat, 2))
            
            else:  # Cat's turn
                for prev_cat in graph[cat]:
                    if prev_cat == 0:  # Cat can't go to hole
                        continue
                    
                    if dp[mouse][prev_cat][1] == 0:
                        if dp[mouse][cat][2] == 2:  # Cat wins
                            dp[mouse][prev_cat][1] = 2
                            queue.append((mouse, prev_cat, 1))
                        else:
                            degree[mouse][prev_cat][1] -= 1
                            if degree[mouse][prev_cat][1] == 0:
                                dp[mouse][prev_cat][1] = 1
                                queue.append((mouse, prev_cat, 1))
        
        return dp[1][2][1]  # Mouse starts at 1, cat starts at 2, mouse moves first

# ==================== GRUNDY NUMBERS ====================

class GrundyNumbers:
    """
    Grundy Number Computations
    
    Implementation of Sprague-Grundy theorem for impartial games.
    """
    
    def compute_grundy_number(self, n: int, moves: List[int]) -> int:
        """
        Compute Grundy number for a position
        
        Time Complexity: O(n * len(moves))
        Space Complexity: O(n)
        
        Args:
            n: Current position/state
            moves: List of possible moves (reductions) from any position
        
        Returns:
            Grundy number for position n
        """
        grundy = [0] * (n + 1)
        
        for i in range(1, n + 1):
            reachable = set()
            
            for move in moves:
                if i >= move:
                    reachable.add(grundy[i - move])
            
            # Find mex (minimum excludant)
            mex = 0
            while mex in reachable:
                mex += 1
            
            grundy[i] = mex
        
        return grundy[n]
    
    def grundy_for_subtraction_game(self, n: int, subtraction_set: Set[int]) -> int:
        """
        Grundy numbers for subtraction games
        
        Args:
            n: Position
            subtraction_set: Set of allowed subtractions
        
        Returns:
            Grundy number
        """
        grundy = [0] * (n + 1)
        
        for i in range(1, n + 1):
            reachable = set()
            
            for sub in subtraction_set:
                if i >= sub:
                    reachable.add(grundy[i - sub])
            
            mex = 0
            while mex in reachable:
                mex += 1
            
            grundy[i] = mex
        
        return grundy[n]
    
    def combine_games_grundy(self, grundy_values: List[int]) -> int:
        """
        Combine multiple independent games using Sprague-Grundy theorem
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            grundy_values: List of Grundy numbers for individual games
        
        Returns:
            Combined Grundy number (XOR of all values)
        """
        result = 0
        for g in grundy_values:
            result ^= g
        
        return result
    
    def nim_sum_analysis(self, piles: List[int]) -> Dict[str, any]:
        """
        Comprehensive analysis of Nim position
        
        Args:
            piles: List of pile sizes
        
        Returns:
            Dictionary with analysis results
        """
        nim_sum = 0
        for pile in piles:
            nim_sum ^= pile
        
        analysis = {
            'nim_sum': nim_sum,
            'is_winning': nim_sum != 0,
            'winning_moves': []
        }
        
        if nim_sum != 0:  # Winning position
            for i, pile in enumerate(piles):
                target = pile ^ nim_sum
                if target < pile:
                    analysis['winning_moves'].append({
                        'pile_index': i,
                        'from': pile,
                        'to': target,
                        'remove': pile - target
                    })
        
        return analysis

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different game theory approaches"""
    print("=== Game Theory DP Performance Analysis ===\n")
    
    import random
    
    # Test Stone Game variants
    test_sizes = [10, 20, 30]
    
    for size in test_sizes:
        piles = [random.randint(1, 100) for _ in range(size)]
        
        print(f"Stone Game with {size} piles:")
        
        stone_game = StoneGameVariants()
        
        start_time = time.time()
        result_i = stone_game.stone_game_i(piles)
        time_i = time.time() - start_time
        
        print(f"  Stone Game I: {result_i} ({time_i:.6f}s)")
        
        # Test Nim Game
        nim_game = NimGameVariants()
        
        start_time = time.time()
        nim_result = nim_game.nim_game_general(piles)
        time_nim = time.time() - start_time
        
        print(f"  Nim Game: {nim_result} ({time_nim:.6f}s)")
        print()

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Game Theory DP Demo ===\n")
    
    # Stone Game Variants
    print("1. Stone Game Variants:")
    stone_game = StoneGameVariants()
    
    piles = [5, 3, 4, 5]
    
    result_i = stone_game.stone_game_i(piles)
    print(f"  Stone Game I with piles {piles}: Alex wins = {result_i}")
    
    result_ii = stone_game.stone_game_ii(piles)
    print(f"  Stone Game II: Alex gets {result_ii} stones")
    
    stone_values = [1, 2, 3, 7]
    result_iii = stone_game.stone_game_iii(stone_values)
    print(f"  Stone Game III with values {stone_values}: Winner = {result_iii}")
    
    n_stones = 7
    result_iv = stone_game.stone_game_iv(n_stones)
    print(f"  Stone Game IV with {n_stones} stones: Alice wins = {result_iv}")
    
    stone_values_v = [6, 2, 3, 4, 5, 5]
    result_v = stone_game.stone_game_v(stone_values_v)
    print(f"  Stone Game V: Alice's max score = {result_v}")
    print()
    
    # Nim Game Variants
    print("2. Nim Game Variants:")
    nim_game = NimGameVariants()
    
    # Basic Nim
    n = 4
    basic_nim = nim_game.nim_game_basic(n)
    print(f"  Basic Nim with {n} stones: First player wins = {basic_nim}")
    
    # General Nim
    piles = [1, 3, 5, 7]
    general_nim = nim_game.nim_game_general(piles)
    print(f"  General Nim with piles {piles}: First player wins = {general_nim}")
    
    # Misère Nim
    misere_nim = nim_game.nim_game_misere(piles)
    print(f"  Misère Nim with piles {piles}: First player wins = {misere_nim}")
    
    # Kayles
    pins = 8
    kayles_result = nim_game.nim_game_kayles(pins)
    print(f"  Kayles with {pins} pins: First player wins = {kayles_result}")
    
    # Nim with constraints
    max_take = [2, 3, 2, 4]  # Max take for each pile
    constrained_nim = nim_game.nim_game_with_constraints(piles, max_take)
    print(f"  Constrained Nim: First player wins = {constrained_nim}")
    print()
    
    # General Game Theory
    print("3. General Game Theory Problems:")
    general_game = GeneralGameTheory()
    
    # Can I Win
    max_choosable = 10
    desired_total = 11
    can_win = general_game.can_i_win(max_choosable, desired_total)
    print(f"  Can I Win (max={max_choosable}, target={desired_total}): {can_win}")
    
    # Predict the Winner
    nums = [1, 5, 233, 7]
    predict_winner = general_game.predict_the_winner_general(nums)
    print(f"  Predict Winner with {nums}: First player wins = {predict_winner}")
    
    # Flip Game II
    state = "++++"
    flip_game = general_game.flip_game_ii(state)
    print(f"  Flip Game II with '{state}': First player wins = {flip_game}")
    print()
    
    # Grundy Numbers
    print("4. Grundy Numbers:")
    grundy = GrundyNumbers()
    
    # Subtraction game
    n = 10
    moves = [1, 2, 3]
    grundy_val = grundy.compute_grundy_number(n, moves)
    print(f"  Grundy number for position {n} with moves {moves}: {grundy_val}")
    
    # Subtraction game with set
    subtraction_set = {1, 3, 4}
    grundy_sub = grundy.grundy_for_subtraction_game(n, subtraction_set)
    print(f"  Subtraction game Grundy number: {grundy_sub}")
    
    # Combine games
    game_grundys = [2, 3, 1]
    combined = grundy.combine_games_grundy(game_grundys)
    print(f"  Combined Grundy for games {game_grundys}: {combined}")
    
    # Nim sum analysis
    nim_piles = [3, 5, 7]
    analysis = grundy.nim_sum_analysis(nim_piles)
    print(f"  Nim analysis for {nim_piles}:")
    print(f"    Nim sum: {analysis['nim_sum']}")
    print(f"    Is winning: {analysis['is_winning']}")
    print(f"    Winning moves: {analysis['winning_moves']}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("=== Game Theory DP Pattern Recognition ===")
    print("Common Game Theory Patterns:")
    print("  1. Zero-sum games: One player's gain = other's loss")
    print("  2. Impartial games: Same moves available to both players")
    print("  3. Partisan games: Different moves for different players")
    print("  4. Minimax: Maximize own score, minimize opponent's")
    print("  5. Grundy numbers: For combining independent games")
    
    print("\nKey Concepts:")
    print("  1. Winning/Losing positions: N-positions vs P-positions")
    print("  2. Strategy stealing: If move is good for opponent, do it yourself")
    print("  3. Sprague-Grundy theorem: XOR of Grundy numbers")
    print("  4. Mex function: Minimum excludant")
    print("  5. Game equivalence: Games with same Grundy number")
    
    print("\nDP State Definition:")
    print("  1. dp[state] = optimal value from current state")
    print("  2. dp[i][j] = optimal value for range [i, j]")
    print("  3. dp[mask] = optimal value for bitmask state")
    print("  4. dp[pos][turn] = value at position pos, player turn")
    
    print("\nSolution Strategies:")
    print("  1. Backward induction: Work from terminal positions")
    print("  2. Memoization: Cache computed game states")
    print("  3. Mathematical insights: Look for patterns/formulas")
    print("  4. Grundy analysis: For multi-pile/multi-game scenarios")
    
    print("\nReal-world Applications:")
    print("  1. Economics: Auction theory, bargaining")
    print("  2. Computer Science: AI game playing, algorithm design")
    print("  3. Military: Strategic planning, resource allocation")
    print("  4. Biology: Evolution, population dynamics")
    print("  5. Political Science: Voting systems, coalition formation")
    
    print("\nCommon Pitfalls:")
    print("  1. Forgetting about ties/draws")
    print("  2. Incorrect terminal condition handling")
    print("  3. Not considering optimal play assumption")
    print("  4. Misunderstanding game rules (normal vs misère)")
    print("  5. Exponential state space without proper pruning")
    
    print("\n=== Demo Complete ===") 