"""
Dynamic Programming - Interval Patterns
This module implements various DP problems on intervals including burst balloons,
merge stones, predict the winner, and optimal interval partitioning strategies.
"""

from typing import List, Dict, Tuple, Optional
import time
from functools import lru_cache

# ==================== BURST BALLOONS ====================

class BurstBalloons:
    """
    Burst Balloons Problem
    
    LeetCode 312 - Burst Balloons
    Find maximum coins by bursting balloons optimally.
    Key insight: Think about which balloon to burst LAST in each interval.
    """
    
    def max_coins_tabulation(self, nums: List[int]) -> int:
        """
        Tabulation approach for burst balloons
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            nums: Array of balloon values
        
        Returns:
            Maximum coins obtainable
        """
        if not nums:
            return 0
        
        # Add dummy balloons with value 1 at both ends
        balloons = [1] + nums + [1]
        n = len(balloons)
        
        # dp[i][j] = max coins from bursting balloons between i and j (exclusive)
        dp = [[0] * n for _ in range(n)]
        
        # Length of gap between i and j
        for length in range(2, n):
            for i in range(n - length):
                j = i + length
                
                # Try bursting each balloon k last in interval (i, j)
                for k in range(i + 1, j):
                    coins = (balloons[i] * balloons[k] * balloons[j] + 
                           dp[i][k] + dp[k][j])
                    dp[i][j] = max(dp[i][j], coins)
        
        return dp[0][n - 1]
    
    def max_coins_memoization(self, nums: List[int]) -> int:
        """
        Memoization approach for burst balloons
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        if not nums:
            return 0
        
        balloons = [1] + nums + [1]
        memo = {}
        
        def burst(left: int, right: int) -> int:
            if left + 1 == right:
                return 0
            
            if (left, right) in memo:
                return memo[(left, right)]
            
            max_coins = 0
            for k in range(left + 1, right):
                coins = (balloons[left] * balloons[k] * balloons[right] + 
                        burst(left, k) + burst(k, right))
                max_coins = max(max_coins, coins)
            
            memo[(left, right)] = max_coins
            return max_coins
        
        return burst(0, len(balloons) - 1)
    
    def max_coins_with_sequence(self, nums: List[int]) -> Tuple[int, List[int]]:
        """
        Find maximum coins and the order of bursting
        
        Args:
            nums: Array of balloon values
        
        Returns:
            Tuple of (max_coins, burst_order)
        """
        if not nums:
            return 0, []
        
        balloons = [1] + nums + [1]
        n = len(balloons)
        
        dp = [[0] * n for _ in range(n)]
        choice = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for i in range(n - length):
                j = i + length
                
                for k in range(i + 1, j):
                    coins = (balloons[i] * balloons[k] * balloons[j] + 
                           dp[i][k] + dp[k][j])
                    
                    if coins > dp[i][j]:
                        dp[i][j] = coins
                        choice[i][j] = k
        
        # Reconstruct the order
        burst_order = []
        
        def reconstruct(left: int, right: int):
            if left + 1 >= right:
                return
            
            k = choice[left][right]
            burst_order.append(nums[k - 1])  # Convert back to original index
            
            reconstruct(left, k)
            reconstruct(k, right)
        
        reconstruct(0, n - 1)
        return dp[0][n - 1], burst_order
    
    def max_coins_with_constraints(self, nums: List[int], 
                                 forbidden_pairs: List[Tuple[int, int]]) -> int:
        """
        Burst balloons with constraints on which balloons cannot be adjacent
        
        Args:
            nums: Array of balloon values
            forbidden_pairs: Pairs of balloon indices that cannot be adjacent
        
        Returns:
            Maximum coins with constraints
        """
        if not nums:
            return 0
        
        balloons = [1] + nums + [1]
        n = len(balloons)
        
        # Convert forbidden pairs to set for O(1) lookup
        forbidden = set()
        for i, j in forbidden_pairs:
            forbidden.add((i + 1, j + 1))  # Adjust for dummy balloons
            forbidden.add((j + 1, i + 1))
        
        memo = {}
        
        def burst(left: int, right: int) -> int:
            if left + 1 == right:
                return 0
            
            if (left, right) in memo:
                return memo[(left, right)]
            
            max_coins = 0
            for k in range(left + 1, right):
                # Check if bursting k creates forbidden adjacency
                valid = True
                for other in range(left + 1, right):
                    if other != k and (k, other) in forbidden:
                        valid = False
                        break
                
                if valid:
                    coins = (balloons[left] * balloons[k] * balloons[right] + 
                           burst(left, k) + burst(k, right))
                    max_coins = max(max_coins, coins)
            
            memo[(left, right)] = max_coins
            return max_coins
        
        return burst(0, n - 1)

# ==================== MERGE STONES ====================

class MergeStones:
    """
    Merge Stones Problems
    
    LeetCode 1000 - Minimum Cost to Merge Stones
    Merge stones optimally to minimize cost.
    """
    
    def merge_cost_k_piles(self, stones: List[int], k: int) -> int:
        """
        Minimum cost to merge stones into one pile with k-way merges
        
        LeetCode 1000 - Minimum Cost to Merge Stones
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            stones: Array of stone weights
            k: Number of piles that can be merged at once
        
        Returns:
            Minimum cost to merge all stones, -1 if impossible
        """
        n = len(stones)
        
        # Check if it's possible to merge
        if (n - 1) % (k - 1) != 0:
            return -1
        
        # Prefix sums for quick range sum calculation
        prefix_sum = [0]
        for stone in stones:
            prefix_sum.append(prefix_sum[-1] + stone)
        
        def range_sum(i: int, j: int) -> int:
            return prefix_sum[j + 1] - prefix_sum[i]
        
        # dp[i][j][p] = min cost to merge stones[i:j+1] into p piles
        dp = [[[float('inf')] * (k + 1) for _ in range(n)] for _ in range(n)]
        
        # Base case: single stone is already 1 pile with 0 cost
        for i in range(n):
            dp[i][i][1] = 0
        
        # Fill DP table
        for length in range(2, n + 1):  # interval length
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Merge into p piles (2 <= p <= k)
                for p in range(2, k + 1):
                    for mid in range(i, j, k - 1):
                        dp[i][j][p] = min(dp[i][j][p],
                                        dp[i][mid][1] + dp[mid + 1][j][p - 1])
                
                # Merge k piles into 1 pile
                dp[i][j][1] = dp[i][j][k] + range_sum(i, j)
        
        return dp[0][n - 1][1] if dp[0][n - 1][1] != float('inf') else -1
    
    def merge_cost_with_sequence(self, stones: List[int], k: int) -> Tuple[int, List[List[int]]]:
        """
        Find minimum cost and the actual merge sequence
        
        Args:
            stones: Array of stone weights
            k: Number of piles in each merge
        
        Returns:
            Tuple of (min_cost, merge_sequence)
        """
        n = len(stones)
        if (n - 1) % (k - 1) != 0:
            return -1, []
        
        prefix_sum = [0]
        for stone in stones:
            prefix_sum.append(prefix_sum[-1] + stone)
        
        def range_sum(i: int, j: int) -> int:
            return prefix_sum[j + 1] - prefix_sum[i]
        
        dp = [[[float('inf')] * (k + 1) for _ in range(n)] for _ in range(n)]
        merge_points = [[[[] for _ in range(k + 1)] for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            dp[i][i][1] = 0
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                for p in range(2, k + 1):
                    for mid in range(i, j, k - 1):
                        cost = dp[i][mid][1] + dp[mid + 1][j][p - 1]
                        if cost < dp[i][j][p]:
                            dp[i][j][p] = cost
                            merge_points[i][j][p] = [mid]
                
                if dp[i][j][k] != float('inf'):
                    dp[i][j][1] = dp[i][j][k] + range_sum(i, j)
                    merge_points[i][j][1] = merge_points[i][j][k]
        
        merge_sequence = []
        
        def reconstruct_merges(i: int, j: int, p: int):
            if i == j:
                return
            
            if p == 1:
                # This is where we do the actual merge
                merge_sequence.append(stones[i:j + 1])
                reconstruct_merges(i, j, k)
            else:
                # Split into smaller parts
                for mid in merge_points[i][j][p]:
                    reconstruct_merges(i, mid, 1)
                    reconstruct_merges(mid + 1, j, p - 1)
        
        if dp[0][n - 1][1] != float('inf'):
            reconstruct_merges(0, n - 1, 1)
        
        return dp[0][n - 1][1] if dp[0][n - 1][1] != float('inf') else -1, merge_sequence
    
    def min_cost_merge_files(self, files: List[int]) -> int:
        """
        Minimum cost to merge files (special case of k=2)
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            files: Array of file sizes
        
        Returns:
            Minimum cost to merge all files
        """
        n = len(files)
        if n <= 1:
            return 0
        
        # Prefix sums
        prefix_sum = [0]
        for file_size in files:
            prefix_sum.append(prefix_sum[-1] + file_size)
        
        def range_sum(i: int, j: int) -> int:
            return prefix_sum[j + 1] - prefix_sum[i]
        
        # dp[i][j] = min cost to merge files[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + range_sum(i, j)
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n - 1]

# ==================== PREDICT THE WINNER ====================

class PredictTheWinner:
    """
    Game Theory with Intervals
    
    LeetCode 486 - Predict the Winner
    Two players optimally pick from ends of array.
    """
    
    def predict_winner_dp(self, nums: List[int]) -> bool:
        """
        Determine if Player 1 can win the game
        
        LeetCode 486 - Predict the Winner
        
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        
        Args:
            nums: Array of numbers
        
        Returns:
            True if Player 1 can win or tie
        """
        n = len(nums)
        
        # dp[i][j] = max score difference (player1 - player2) for nums[i:j+1]
        dp = [[0] * n for _ in range(n)]
        
        # Base case: single element
        for i in range(n):
            dp[i][i] = nums[i]
        
        # Fill DP table
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                # Player picks from left (nums[i])
                pick_left = nums[i] - dp[i + 1][j]
                
                # Player picks from right (nums[j])
                pick_right = nums[j] - dp[i][j - 1]
                
                dp[i][j] = max(pick_left, pick_right)
        
        return dp[0][n - 1] >= 0
    
    def predict_winner_with_moves(self, nums: List[int]) -> Tuple[bool, List[Tuple[int, str]]]:
        """
        Predict winner and return the optimal moves
        
        Args:
            nums: Array of numbers
        
        Returns:
            Tuple of (can_player1_win, optimal_moves)
        """
        n = len(nums)
        dp = [[0] * n for _ in range(n)]
        move = [[None] * n for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = nums[i]
            move[i][i] = 'single'
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                pick_left = nums[i] - dp[i + 1][j]
                pick_right = nums[j] - dp[i][j - 1]
                
                if pick_left >= pick_right:
                    dp[i][j] = pick_left
                    move[i][j] = 'left'
                else:
                    dp[i][j] = pick_right
                    move[i][j] = 'right'
        
        # Reconstruct moves
        moves = []
        
        def get_moves(i: int, j: int, is_player1: bool):
            if i > j:
                return
            
            if i == j:
                moves.append((nums[i], 'Player1' if is_player1 else 'Player2'))
                return
            
            if move[i][j] == 'left':
                moves.append((nums[i], 'Player1' if is_player1 else 'Player2'))
                get_moves(i + 1, j, not is_player1)
            else:
                moves.append((nums[j], 'Player1' if is_player1 else 'Player2'))
                get_moves(i, j - 1, not is_player1)
        
        get_moves(0, n - 1, True)
        return dp[0][n - 1] >= 0, moves
    
    def stone_game_variants(self, piles: List[int], variant: str = 'basic') -> bool:
        """
        Various Stone Game variants
        
        Args:
            piles: Array of stone piles
            variant: Type of stone game ('basic', 'odd_even', 'alternating')
        
        Returns:
            True if Alice (first player) wins
        """
        n = len(piles)
        
        if variant == 'basic':
            # LeetCode 877 - Stone Game (Alice always wins with even n)
            return True
        
        elif variant == 'odd_even':
            # Players can only pick from odd/even positions alternately
            dp = {}
            
            def solve(i: int, j: int, turn: int, can_pick_odd: bool) -> int:
                if i > j:
                    return 0
                
                if (i, j, turn, can_pick_odd) in dp:
                    return dp[(i, j, turn, can_pick_odd)]
                
                result = float('-inf') if turn == 0 else float('inf')
                
                # Try picking from left
                if (i % 2 == 1) == can_pick_odd:  # Can pick from position i
                    if turn == 0:  # Alice's turn (maximize)
                        result = max(result, piles[i] + solve(i + 1, j, 1, not can_pick_odd))
                    else:  # Bob's turn (minimize Alice's score)
                        result = min(result, solve(i + 1, j, 0, not can_pick_odd))
                
                # Try picking from right
                if (j % 2 == 1) == can_pick_odd:  # Can pick from position j
                    if turn == 0:  # Alice's turn
                        result = max(result, piles[j] + solve(i, j - 1, 1, not can_pick_odd))
                    else:  # Bob's turn
                        result = min(result, solve(i, j - 1, 0, not can_pick_odd))
                
                dp[(i, j, turn, can_pick_odd)] = result
                return result
            
            alice_score = solve(0, n - 1, 0, True)
            total_sum = sum(piles)
            return alice_score > total_sum - alice_score
        
        else:  # alternating
            # Standard predict the winner logic
            return self.predict_winner_dp(piles)

# ==================== ADVANCED INTERVAL PROBLEMS ====================

class AdvancedIntervalDP:
    """
    Advanced interval DP problems with complex constraints
    """
    
    def optimal_binary_search_tree(self, keys: List[int], frequencies: List[int]) -> int:
        """
        Construct optimal binary search tree
        
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        
        Args:
            keys: Sorted array of keys
            frequencies: Access frequency of each key
        
        Returns:
            Minimum expected search cost
        """
        n = len(keys)
        if n != len(frequencies):
            raise ValueError("Keys and frequencies must have same length")
        
        # dp[i][j] = min cost for keys[i:j+1]
        dp = [[0] * n for _ in range(n)]
        # sum_freq[i][j] = sum of frequencies from i to j
        sum_freq = [[0] * n for _ in range(n)]
        
        # Initialize frequency sums
        for i in range(n):
            sum_freq[i][i] = frequencies[i]
            for j in range(i + 1, n):
                sum_freq[i][j] = sum_freq[i][j - 1] + frequencies[j]
        
        # Base case: single keys
        for i in range(n):
            dp[i][i] = frequencies[i]
        
        # Fill for intervals of length 2, 3, ...
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                # Try each key as root
                for k in range(i, j + 1):
                    left_cost = dp[i][k - 1] if k > i else 0
                    right_cost = dp[k + 1][j] if k < j else 0
                    cost = left_cost + right_cost + sum_freq[i][j]
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n - 1]
    
    def palindrome_partitioning_cost(self, s: str, costs: List[int]) -> int:
        """
        Minimum cost palindrome partitioning with custom costs
        
        Args:
            s: Input string
            costs: Cost of making each cut
        
        Returns:
            Minimum cost to partition into palindromes
        """
        n = len(s)
        
        # Precompute palindrome check
        is_palindrome = [[False] * n for _ in range(n)]
        
        for i in range(n):
            is_palindrome[i][i] = True
        
        for i in range(n - 1):
            is_palindrome[i][i + 1] = (s[i] == s[i + 1])
        
        for length in range(3, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
        
        # dp[i] = min cost for s[0:i+1]
        dp = [float('inf')] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0
            else:
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        dp[i] = min(dp[i], dp[j] + costs[j])
        
        return dp[n - 1]
    
    def matrix_chain_variants(self, dimensions: List[int], 
                            variant: str = 'basic') -> int:
        """
        Various matrix chain multiplication variants
        
        Args:
            dimensions: Matrix dimensions array
            variant: Type of problem ('basic', 'parallel', 'memory_limited')
        
        Returns:
            Optimal cost based on variant
        """
        n = len(dimensions) - 1
        if n <= 1:
            return 0
        
        if variant == 'basic':
            # Standard matrix chain multiplication
            dp = [[0] * n for _ in range(n)]
            
            for length in range(2, n + 1):
                for i in range(n - length + 1):
                    j = i + length - 1
                    dp[i][j] = float('inf')
                    
                    for k in range(i, j):
                        cost = (dp[i][k] + dp[k + 1][j] + 
                               dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                        dp[i][j] = min(dp[i][j], cost)
            
            return dp[0][n - 1]
        
        elif variant == 'parallel':
            # Matrix chain with parallel processing capability
            dp = [[0] * n for _ in range(n)]
            
            for length in range(2, n + 1):
                for i in range(n - length + 1):
                    j = i + length - 1
                    dp[i][j] = float('inf')
                    
                    for k in range(i, j):
                        # Parallel cost: max of left and right + merge cost
                        parallel_cost = (max(dp[i][k], dp[k + 1][j]) + 
                                       dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                        dp[i][j] = min(dp[i][j], parallel_cost)
            
            return dp[0][n - 1]
        
        else:  # memory_limited
            # Matrix chain with memory constraints
            memory_limit = 1000000  # Example limit
            dp = {}
            
            def solve(i: int, j: int, memory_used: int) -> int:
                if i >= j:
                    return 0
                
                if (i, j, memory_used) in dp:
                    return dp[(i, j, memory_used)]
                
                result = float('inf')
                
                for k in range(i, j):
                    # Calculate memory needed for intermediate result
                    intermediate_memory = dimensions[i] * dimensions[j + 1]
                    
                    if memory_used + intermediate_memory <= memory_limit:
                        cost = (solve(i, k, memory_used) + 
                               solve(k + 1, j, memory_used + intermediate_memory) +
                               dimensions[i] * dimensions[k + 1] * dimensions[j + 1])
                        result = min(result, cost)
                
                dp[(i, j, memory_used)] = result
                return result
            
            return solve(0, n - 1, 0)
    
    def rod_cutting_with_constraints(self, length: int, prices: List[int], 
                                   max_cuts: int) -> int:
        """
        Rod cutting with maximum number of cuts constraint
        
        Args:
            length: Length of rod
            prices: Price array where prices[i] is price of rod of length i+1
            max_cuts: Maximum number of cuts allowed
        
        Returns:
            Maximum revenue with cut constraint
        """
        # dp[i][j] = max revenue for rod of length i with at most j cuts
        dp = [[0] * (max_cuts + 1) for _ in range(length + 1)]
        
        for i in range(1, length + 1):
            for j in range(max_cuts + 1):
                # Don't cut (use full length)
                if i <= len(prices):
                    dp[i][j] = max(dp[i][j], prices[i - 1])
                
                # Try cutting at each position
                if j > 0:
                    for cut_pos in range(1, i):
                        if cut_pos <= len(prices):
                            dp[i][j] = max(dp[i][j], 
                                         prices[cut_pos - 1] + dp[i - cut_pos][j - 1])
        
        return dp[length][max_cuts]

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different interval DP approaches"""
    print("=== Interval DP Performance Analysis ===\n")
    
    import random
    
    # Test Burst Balloons
    test_sizes = [10, 15, 20]
    
    for size in test_sizes:
        balloons = [random.randint(1, 10) for _ in range(size)]
        
        print(f"Burst Balloons with {size} balloons:")
        
        burst = BurstBalloons()
        
        start_time = time.time()
        result_tab = burst.max_coins_tabulation(balloons)
        time_tab = time.time() - start_time
        
        start_time = time.time()
        result_memo = burst.max_coins_memoization(balloons)
        time_memo = time.time() - start_time
        
        print(f"  Tabulation: {result_tab} ({time_tab:.6f}s)")
        print(f"  Memoization: {result_memo} ({time_memo:.6f}s)")
        print(f"  Results match: {result_tab == result_memo}")
        print()

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Interval DP Demo ===\n")
    
    # Burst Balloons
    print("1. Burst Balloons Problems:")
    burst = BurstBalloons()
    
    balloons = [3, 1, 5, 8]
    max_coins_tab = burst.max_coins_tabulation(balloons)
    max_coins_memo = burst.max_coins_memoization(balloons)
    max_coins_seq, burst_order = burst.max_coins_with_sequence(balloons)
    
    print(f"  Balloons: {balloons}")
    print(f"  Maximum coins (tabulation): {max_coins_tab}")
    print(f"  Maximum coins (memoization): {max_coins_memo}")
    print(f"  Maximum coins with sequence: {max_coins_seq}")
    print(f"  Burst order: {burst_order}")
    
    # With constraints
    forbidden_pairs = [(0, 2)]
    constrained_coins = burst.max_coins_with_constraints(balloons, forbidden_pairs)
    print(f"  With constraints {forbidden_pairs}: {constrained_coins}")
    print()
    
    # Merge Stones
    print("2. Merge Stones Problems:")
    merge = MergeStones()
    
    stones = [3, 2, 4, 1]
    k = 2
    min_cost = merge.merge_cost_k_piles(stones, k)
    min_cost_seq, merge_sequence = merge.merge_cost_with_sequence(stones, k)
    
    print(f"  Stones: {stones}, k={k}")
    print(f"  Minimum merge cost: {min_cost}")
    print(f"  Merge sequence: {merge_sequence}")
    
    # Merge files (k=2)
    files = [20, 30, 10, 40]
    file_cost = merge.min_cost_merge_files(files)
    print(f"  File merge cost {files}: {file_cost}")
    print()
    
    # Predict the Winner
    print("3. Predict the Winner Problems:")
    predictor = PredictTheWinner()
    
    nums = [1, 5, 2, 4, 6]
    can_win = predictor.predict_winner_dp(nums)
    can_win_moves, moves = predictor.predict_winner_with_moves(nums)
    
    print(f"  Array: {nums}")
    print(f"  Player 1 can win: {can_win}")
    print(f"  Optimal moves: {moves}")
    
    # Stone game variants
    piles = [5, 3, 4, 5]
    basic_win = predictor.stone_game_variants(piles, 'basic')
    alternating_win = predictor.stone_game_variants(piles, 'alternating')
    
    print(f"  Stone piles: {piles}")
    print(f"  Basic stone game (Alice wins): {basic_win}")
    print(f"  Alternating picks (Alice wins): {alternating_win}")
    print()
    
    # Advanced Interval Problems
    print("4. Advanced Interval Problems:")
    advanced = AdvancedIntervalDP()
    
    # Optimal BST
    keys = [10, 20, 30]
    frequencies = [34, 8, 50]
    bst_cost = advanced.optimal_binary_search_tree(keys, frequencies)
    print(f"  Optimal BST cost for keys {keys} with freq {frequencies}: {bst_cost}")
    
    # Palindrome partitioning with costs
    string = "aab"
    costs = [1, 2]  # Cost of cuts after position 0 and 1
    palindrome_cost = advanced.palindrome_partitioning_cost(string, costs)
    print(f"  Palindrome partition cost for '{string}' with costs {costs}: {palindrome_cost}")
    
    # Matrix chain variants
    dimensions = [40, 20, 30, 10, 30]
    basic_mcm = advanced.matrix_chain_variants(dimensions, 'basic')
    parallel_mcm = advanced.matrix_chain_variants(dimensions, 'parallel')
    
    print(f"  Matrix chain {len(dimensions)-1} matrices:")
    print(f"    Basic MCM: {basic_mcm}")
    print(f"    Parallel MCM: {parallel_mcm}")
    
    # Rod cutting with constraints
    rod_length = 8
    rod_prices = [1, 5, 8, 9, 10, 17, 17, 20]
    max_cuts = 3
    constrained_revenue = advanced.rod_cutting_with_constraints(rod_length, rod_prices, max_cuts)
    print(f"  Rod cutting (length {rod_length}, max {max_cuts} cuts): {constrained_revenue}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("=== Interval DP Pattern Recognition ===")
    print("Common Interval DP Patterns:")
    print("  1. Range DP: dp[i][j] represents optimal solution for range [i, j]")
    print("  2. Last operation: Think about what to do LAST in the interval")
    print("  3. Split point: Try all possible ways to split interval [i, j]")
    print("  4. Merge operations: Combine adjacent intervals optimally")
    print("  5. Game theory: Alternating players making optimal moves")
    
    print("\nKey Insight - Think About:")
    print("  1. What's the last balloon to burst?")
    print("  2. What's the root of BST for this range?")
    print("  3. Where to make the last merge/cut?")
    print("  4. Which element to pick last?")
    print("  5. What's the optimal split point?")
    
    print("\nState Transition Pattern:")
    print("  for length in range(2, n+1):")
    print("    for i in range(n-length+1):")
    print("      j = i + length - 1")
    print("      for k in range(i, j):  # Split/decision point")
    print("        dp[i][j] = optimize(dp[i][k], dp[k+1][j], cost(i,k,j))")
    
    print("\nProblem Types:")
    print("  1. Bursting/Removing: What to remove last?")
    print("  2. Merging/Combining: How to combine optimally?")
    print("  3. Game Theory: Optimal play by alternating players")
    print("  4. Tree Construction: Optimal tree structure")
    print("  5. Partitioning: How to partition optimally?")
    
    print("\nReal-world Applications:")
    print("  1. Compiler optimization (expression evaluation)")
    print("  2. Database query optimization")
    print("  3. Game AI and decision making")
    print("  4. Resource allocation and scheduling")
    print("  5. Network protocol optimization")
    print("  6. Financial portfolio optimization")
    
    print("\nOptimization Techniques:")
    print("  1. Knuth-Yao speedup for quadrangle inequality")
    print("  2. Divide and conquer optimization")
    print("  3. Convex hull optimization")
    print("  4. Memoization vs tabulation trade-offs")
    
    print("\n=== Demo Complete ===") 