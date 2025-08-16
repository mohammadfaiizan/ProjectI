"""
LeetCode 688: Knight Probability in Chessboard
Difficulty: Medium
Category: Probability DP - Markov Chain Analysis

PROBLEM DESCRIPTION:
===================
On an n x n chessboard, a knight starts at the cell (row, column) and attempts to make exactly k moves.
The rows and columns are 0-indexed, so the top-left cell is (0, 0), and the bottom-right cell is (n-1, n-1).

A chess knight has 8 possible moves it can make, as illustrated below. Each move is two cells in a cardinal direction, then one cell in an orthogonal direction.

Return the probability that the knight remains on the chessboard after it has stopped making moves.

Example 1:
Input: n = 3, k = 2, row = 0, column = 0
Output: 0.0625
Explanation: There are two moves (to (1,2), (2,1)) and four possible paths off the board:
(0,0) -> (2,1) -> (0,2) [off the board]
(0,0) -> (2,1) -> (1,3) [off the board]
(0,0) -> (1,2) -> (3,0) [off the board]
(0,0) -> (1,2) -> (2,4) [off the board]
And 4 possible paths that stay on the board.

Example 2:
Input: n = 1, k = 0, row = 0, column = 0
Output: 1.0

Constraints:
- 1 <= n <= 25
- 0 <= k <= 100
- 0 <= row, column < n
"""


def knight_probability_recursive(n, k, row, column):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to calculate probability directly.
    
    Time Complexity: O(8^k) - exponential without memoization
    Space Complexity: O(k) - recursion stack
    """
    # Knight moves: 8 possible L-shaped moves
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    
    def is_valid(r, c):
        return 0 <= r < n and 0 <= c < n
    
    def probability(r, c, moves_left):
        # Base case: no moves left
        if moves_left == 0:
            return 1.0 if is_valid(r, c) else 0.0
        
        # If current position is off board, probability is 0
        if not is_valid(r, c):
            return 0.0
        
        total_prob = 0.0
        for dr, dc in directions:
            new_r, new_c = r + dr, c + dc
            total_prob += probability(new_r, new_c, moves_left - 1)
        
        # Each move has equal probability (1/8)
        return total_prob / 8.0
    
    return probability(row, column, k)


def knight_probability_memoization(n, k, row, column):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed states.
    
    Time Complexity: O(n^2 * k) - each state computed once
    Space Complexity: O(n^2 * k) - memoization table
    """
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    memo = {}
    
    def is_valid(r, c):
        return 0 <= r < n and 0 <= c < n
    
    def probability(r, c, moves_left):
        if moves_left == 0:
            return 1.0 if is_valid(r, c) else 0.0
        
        if not is_valid(r, c):
            return 0.0
        
        if (r, c, moves_left) in memo:
            return memo[(r, c, moves_left)]
        
        total_prob = 0.0
        for dr, dc in directions:
            new_r, new_c = r + dr, c + dc
            total_prob += probability(new_r, new_c, moves_left - 1)
        
        result = total_prob / 8.0
        memo[(r, c, moves_left)] = result
        return result
    
    return probability(row, column, k)


def knight_probability_dp_bottom_up(n, k, row, column):
    """
    BOTTOM-UP DP APPROACH:
    =====================
    Use iterative DP building from 0 moves to k moves.
    
    Time Complexity: O(n^2 * k) - three nested loops
    Space Complexity: O(n^2) - two DP tables
    """
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    
    def is_valid(r, c):
        return 0 <= r < n and 0 <= c < n
    
    # dp[r][c] = probability of being at (r,c) after current number of moves
    prev_dp = [[0.0] * n for _ in range(n)]
    curr_dp = [[0.0] * n for _ in range(n)]
    
    # Base case: probability 1 at starting position with 0 moves
    prev_dp[row][column] = 1.0
    
    # For each move
    for move in range(k):
        # Clear current DP table
        for i in range(n):
            for j in range(n):
                curr_dp[i][j] = 0.0
        
        # Calculate probabilities for each cell
        for r in range(n):
            for c in range(n):
                if prev_dp[r][c] > 0:
                    # From (r,c), try all 8 knight moves
                    for dr, dc in directions:
                        new_r, new_c = r + dr, c + dc
                        if is_valid(new_r, new_c):
                            curr_dp[new_r][new_c] += prev_dp[r][c] / 8.0
        
        # Swap tables for next iteration
        prev_dp, curr_dp = curr_dp, prev_dp
    
    # Sum all probabilities on the board
    total_probability = 0.0
    for r in range(n):
        for c in range(n):
            total_probability += prev_dp[r][c]
    
    return total_probability


def knight_probability_optimized(n, k, row, column):
    """
    SPACE-OPTIMIZED DP:
    ==================
    Use only current and previous probability tables.
    
    Time Complexity: O(n^2 * k) - same as bottom-up
    Space Complexity: O(n^2) - optimized space usage
    """
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    
    # Initialize probability table
    dp = [[0.0] * n for _ in range(n)]
    dp[row][column] = 1.0
    
    # For each move
    for _ in range(k):
        new_dp = [[0.0] * n for _ in range(n)]
        
        # Calculate new probabilities
        for r in range(n):
            for c in range(n):
                if dp[r][c] > 0:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            new_dp[nr][nc] += dp[r][c] / 8.0
        
        dp = new_dp
    
    # Sum all probabilities on the board
    return sum(sum(row) for row in dp)


def knight_probability_with_analysis(n, k, row, column):
    """
    KNIGHT PROBABILITY WITH DETAILED ANALYSIS:
    =========================================
    Calculate probability and provide comprehensive insights.
    
    Time Complexity: O(n^2 * k) - DP computation + analysis
    Space Complexity: O(n^2 * k) - DP tables + analysis data
    """
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    
    analysis = {
        'board_size': n,
        'moves': k,
        'start_position': (row, column),
        'move_history': [],
        'probability_evolution': [],
        'final_distribution': None,
        'insights': []
    }
    
    # Track probability evolution over moves
    dp = [[0.0] * n for _ in range(n)]
    dp[row][column] = 1.0
    
    analysis['probability_evolution'].append({
        'move': 0,
        'total_on_board': 1.0,
        'max_cell_prob': 1.0,
        'num_reachable_cells': 1
    })
    
    for move in range(k):
        new_dp = [[0.0] * n for _ in range(n)]
        
        for r in range(n):
            for c in range(n):
                if dp[r][c] > 0:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            new_dp[nr][nc] += dp[r][c] / 8.0
        
        dp = new_dp
        
        # Analyze current state
        total_prob = sum(sum(row) for row in dp)
        max_prob = max(max(row) for row in dp) if total_prob > 0 else 0
        reachable = sum(1 for r in range(n) for c in range(n) if dp[r][c] > 0)
        
        analysis['probability_evolution'].append({
            'move': move + 1,
            'total_on_board': total_prob,
            'max_cell_prob': max_prob,
            'num_reachable_cells': reachable
        })
    
    final_probability = sum(sum(row) for row in dp)
    analysis['final_distribution'] = dp
    
    # Generate insights
    analysis['insights'].append(f"Final probability of staying on board: {final_probability:.6f}")
    analysis['insights'].append(f"Probability of falling off: {1 - final_probability:.6f}")
    
    # Analyze evolution
    evolution = analysis['probability_evolution']
    if len(evolution) > 1:
        initial_reachable = evolution[0]['num_reachable_cells']
        final_reachable = evolution[-1]['num_reachable_cells']
        analysis['insights'].append(f"Reachable cells grew from {initial_reachable} to {final_reachable}")
        
        # Find when probability starts decreasing significantly
        for i in range(1, len(evolution)):
            if evolution[i]['total_on_board'] < 0.5 and evolution[i-1]['total_on_board'] >= 0.5:
                analysis['insights'].append(f"Probability dropped below 50% at move {evolution[i]['move']}")
                break
    
    # Board position analysis
    if n <= 8:  # Only for reasonably small boards
        corner_distance = min(row, column, n-1-row, n-1-column)
        analysis['insights'].append(f"Starting position distance from nearest edge: {corner_distance}")
        
        if corner_distance == 0:
            analysis['insights'].append("Starting at board edge - higher chance of falling off")
        elif corner_distance >= 2:
            analysis['insights'].append("Starting well inside board - better survival chances")
    
    return final_probability, analysis


def knight_probability_analysis(n, k, row, column):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze knight probability with detailed insights.
    """
    print(f"Knight Probability Analysis:")
    print(f"Board size: {n}x{n}")
    print(f"Number of moves: {k}")
    print(f"Starting position: ({row}, {column})")
    print(f"Total possible paths: 8^{k} = {8**k:,}")
    
    if k == 0:
        print("No moves - probability is 1.0")
        return 1.0
    
    # Different approaches
    if k <= 8:  # Only for small k due to exponential complexity
        try:
            recursive = knight_probability_recursive(n, k, row, column)
            print(f"Recursive result: {recursive:.6f}")
        except:
            print("Recursive: Too slow")
    
    memoization = knight_probability_memoization(n, k, row, column)
    bottom_up = knight_probability_dp_bottom_up(n, k, row, column)
    optimized = knight_probability_optimized(n, k, row, column)
    
    print(f"Memoization result: {memoization:.6f}")
    print(f"Bottom-up DP result: {bottom_up:.6f}")
    print(f"Optimized result: {optimized:.6f}")
    
    # Detailed analysis
    detailed_result, analysis = knight_probability_with_analysis(n, k, row, column)
    
    print(f"\nDetailed Analysis:")
    print(f"Final probability: {detailed_result:.6f}")
    
    print(f"\nProbability Evolution:")
    for state in analysis['probability_evolution']:
        print(f"  Move {state['move']:2d}: On board = {state['total_on_board']:.4f}, "
              f"Max cell = {state['max_cell_prob']:.4f}, "
              f"Reachable cells = {state['num_reachable_cells']}")
    
    print(f"\nInsights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    # Show final distribution for small boards
    if n <= 6 and k <= 3:
        print(f"\nFinal Probability Distribution:")
        for r in range(n):
            row_str = "  "
            for c in range(n):
                prob = analysis['final_distribution'][r][c]
                if prob > 0:
                    row_str += f"{prob:.3f} "
                else:
                    row_str += "0.000 "
            print(row_str)
    
    return detailed_result


def knight_probability_variants():
    """
    KNIGHT PROBABILITY VARIANTS:
    ===========================
    Different scenarios and modifications.
    """
    
    def knight_probability_with_obstacles(n, k, row, column, obstacles):
        """Knight probability with obstacles on the board"""
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                      (1, -2), (1, 2), (2, -1), (2, 1)]
        
        obstacle_set = set(obstacles)
        
        def is_valid(r, c):
            return 0 <= r < n and 0 <= c < n and (r, c) not in obstacle_set
        
        memo = {}
        
        def probability(r, c, moves_left):
            if moves_left == 0:
                return 1.0 if is_valid(r, c) else 0.0
            
            if not is_valid(r, c):
                return 0.0
            
            if (r, c, moves_left) in memo:
                return memo[(r, c, moves_left)]
            
            total_prob = 0.0
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                total_prob += probability(new_r, new_c, moves_left - 1)
            
            result = total_prob / 8.0
            memo[(r, c, moves_left)] = result
            return result
        
        return probability(row, column, k)
    
    def knight_probability_different_moves(n, k, row, column, custom_moves):
        """Knight with different possible moves"""
        memo = {}
        
        def is_valid(r, c):
            return 0 <= r < n and 0 <= c < n
        
        def probability(r, c, moves_left):
            if moves_left == 0:
                return 1.0 if is_valid(r, c) else 0.0
            
            if not is_valid(r, c):
                return 0.0
            
            if (r, c, moves_left) in memo:
                return memo[(r, c, moves_left)]
            
            total_prob = 0.0
            for dr, dc in custom_moves:
                new_r, new_c = r + dr, c + dc
                total_prob += probability(new_r, new_c, moves_left - 1)
            
            result = total_prob / len(custom_moves)
            memo[(r, c, moves_left)] = result
            return result
        
        return probability(row, column, k)
    
    def knight_expected_time_to_exit(n, row, column, max_moves=100):
        """Expected number of moves until knight exits board"""
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), 
                      (1, -2), (1, 2), (2, -1), (2, 1)]
        
        memo = {}
        
        def is_valid(r, c):
            return 0 <= r < n and 0 <= c < n
        
        def expected_moves(r, c, moves_so_far):
            if moves_so_far >= max_moves:
                return max_moves  # Cap to prevent infinite recursion
            
            if not is_valid(r, c):
                return moves_so_far  # Exited the board
            
            if (r, c, moves_so_far) in memo:
                return memo[(r, c, moves_so_far)]
            
            total_expected = 0.0
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                total_expected += expected_moves(new_r, new_c, moves_so_far + 1)
            
            result = total_expected / 8.0
            memo[(r, c, moves_so_far)] = result
            return result
        
        return expected_moves(row, column, 0)
    
    # Test variants
    test_cases = [
        (3, 2, 0, 0),
        (3, 2, 1, 1),
        (4, 3, 2, 2),
        (8, 4, 3, 3)
    ]
    
    print("Knight Probability Variants:")
    print("=" * 50)
    
    for n, k, row, col in test_cases:
        print(f"\nBoard: {n}x{n}, Moves: {k}, Start: ({row}, {col})")
        
        basic_prob = knight_probability_optimized(n, k, row, col)
        print(f"Basic knight: {basic_prob:.6f}")
        
        # With obstacles
        if n >= 3:
            obstacles = [(1, 1)] if n > 2 else []
            obstacle_prob = knight_probability_with_obstacles(n, k, row, col, obstacles)
            print(f"With obstacles {obstacles}: {obstacle_prob:.6f}")
        
        # Different moves (rook-like)
        rook_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        rook_prob = knight_probability_different_moves(n, k, row, col, rook_moves)
        print(f"Rook-like moves: {rook_prob:.6f}")
        
        # Expected exit time
        if n <= 4 and k <= 2:
            exit_time = knight_expected_time_to_exit(n, row, col, 20)
            print(f"Expected moves to exit: {exit_time:.2f}")


# Test cases
def test_knight_probability():
    """Test all implementations with various inputs"""
    test_cases = [
        (3, 2, 0, 0, 0.0625),
        (1, 0, 0, 0, 1.0),
        (8, 30, 6, 4, 0.136),  # Approximate
        (3, 1, 1, 1, 1.0),
        (4, 2, 1, 1, 0.75)
    ]
    
    print("Testing Knight Probability Solutions:")
    print("=" * 70)
    
    for i, (n, k, row, col, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}, k = {k}, start = ({row}, {col})")
        print(f"Expected: {expected}")
        
        # Skip recursive for large cases
        if k <= 5:
            try:
                recursive = knight_probability_recursive(n, k, row, col)
                diff = abs(recursive - expected)
                print(f"Recursive:        {recursive:.6f} {'✓' if diff < 0.001 else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = knight_probability_memoization(n, k, row, col)
        bottom_up = knight_probability_dp_bottom_up(n, k, row, col)
        optimized = knight_probability_optimized(n, k, row, col)
        
        memo_diff = abs(memoization - expected)
        dp_diff = abs(bottom_up - expected)
        opt_diff = abs(optimized - expected)
        
        print(f"Memoization:      {memoization:.6f} {'✓' if memo_diff < 0.001 else '✗'}")
        print(f"Bottom-up DP:     {bottom_up:.6f} {'✓' if dp_diff < 0.001 else '✗'}")
        print(f"Optimized:        {optimized:.6f} {'✓' if opt_diff < 0.001 else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    knight_probability_analysis(3, 2, 0, 0)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    knight_probability_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MARKOV CHAIN: Each position represents a state in a Markov chain")
    print("2. TRANSITION PROBABILITIES: 1/8 probability for each valid knight move")
    print("3. ABSORBING STATES: Off-board positions are absorbing (probability 0)")
    print("4. CONVERGENCE: Probability decreases over time due to boundary absorption")
    print("5. SPATIAL DISTRIBUTION: Probability spreads and concentrates over moves")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Random Walk Analysis: Studying bounded random walks with barriers")
    print("• Game Development: Probability calculations for board games")
    print("• Network Analysis: Node survival probability in unreliable networks")
    print("• Physics Simulation: Particle movement with boundary conditions")
    print("• Algorithm Analysis: Randomized algorithm success probability")


if __name__ == "__main__":
    test_knight_probability()


"""
KNIGHT PROBABILITY IN CHESSBOARD - MARKOV CHAIN ANALYSIS:
=========================================================

This problem demonstrates probability DP through Markov chain analysis:
- Random walk on bounded 2D grid with specific transition rules
- Absorbing boundary conditions (off-board = exit)
- Evolution of probability distribution over discrete time steps
- Expected value calculation through state transition analysis

KEY INSIGHTS:
============
1. **MARKOV CHAIN MODELING**: Each board position represents a state in a Markov chain
2. **TRANSITION PROBABILITIES**: Each knight move has equal probability (1/8)
3. **ABSORBING BOUNDARIES**: Off-board positions are absorbing states with probability 0
4. **PROBABILITY CONSERVATION**: Total probability decreases as knight exits board
5. **SPATIAL DISTRIBUTION**: Probability mass spreads and concentrates over time

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(8^k) time, O(k) space
   - Direct probability tree exploration
   - Exponential complexity without memoization

2. **Memoization**: O(n²k) time, O(n²k) space
   - Top-down DP with state caching
   - Each (position, moves_left) computed once

3. **Bottom-up DP**: O(n²k) time, O(n²) space
   - Iterative probability evolution
   - Space-efficient with two probability tables

4. **Optimized DP**: O(n²k) time, O(n²) space
   - Single probability table with in-place updates
   - Most practical implementation

CORE PROBABILITY DP ALGORITHM:
=============================
```python
def knightProbability(n, k, row, column):
    directions = [(-2,-1), (-2,1), (-1,-2), (-1,2), 
                  (1,-2), (1,2), (2,-1), (2,1)]
    
    # dp[r][c] = probability of being at (r,c) after current moves
    dp = [[0.0] * n for _ in range(n)]
    dp[row][column] = 1.0
    
    for _ in range(k):
        new_dp = [[0.0] * n for _ in range(n)]
        
        for r in range(n):
            for c in range(n):
                if dp[r][c] > 0:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            new_dp[nr][nc] += dp[r][c] / 8.0
        
        dp = new_dp
    
    return sum(sum(row) for row in dp)
```

MARKOV CHAIN PROPERTIES:
=======================
**State Space**: All positions (r,c) where 0 ≤ r,c < n
**Transition Matrix**: P[i][j] = probability of moving from state i to state j
**Absorbing States**: All off-board positions (implicit, probability 0)
**Initial Distribution**: π₀ = [0,0,...,1,...,0] (probability 1 at start)

**Evolution**: πₖ = πₖ₋₁ × P (after k steps)

PROBABILITY EVOLUTION ANALYSIS:
==============================
**Conservation**: ∑ᵢⱼ dp[i][j] ≤ 1 (decreases as probability exits)
**Spreading**: Probability mass spreads from initial concentration
**Boundary Effects**: Positions near edges have higher exit probability
**Convergence**: For large k, probability approaches steady state (usually 0)

SPATIAL PROBABILITY PATTERNS:
=============================
**Center Advantage**: Central starting positions have higher survival probability
**Edge Penalty**: Starting near edges increases exit probability
**Corner Disadvantage**: Corner starts have lowest survival rates
**Symmetry**: Symmetric starting positions yield symmetric final distributions

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n²k)
- k iterations of probability evolution
- n² positions to update per iteration
- 8 transitions per position (constant)

**Space Complexity**: O(n²)
- Two probability tables for current and next states
- Can be optimized to single table with careful update order

NUMERICAL CONSIDERATIONS:
========================
**Floating Point Precision**: Use double precision for accurate results
**Probability Normalization**: Verify probability conservation
**Underflow Protection**: Handle very small probabilities for large k
**Convergence Criteria**: Early stopping when probability becomes negligible

APPLICATIONS:
============
- **Random Walk Analysis**: Bounded random walks with absorbing barriers
- **Game Development**: Board game probability calculations
- **Network Reliability**: Node survival in unreliable networks
- **Physics Simulation**: Particle movement with boundary conditions
- **Algorithm Analysis**: Success probability of randomized algorithms

RELATED PROBLEMS:
================
- **Robot in Grid**: Similar random walk problems
- **Brownian Motion**: Continuous analog of discrete random walk
- **Queuing Theory**: State transition probability analysis
- **Monte Carlo Methods**: Simulation-based probability estimation

VARIANTS:
========
- **Different Pieces**: Rook, bishop, or custom movement patterns
- **Obstacles**: Board positions that block movement
- **Non-uniform Transitions**: Weighted move probabilities
- **Multiple Knights**: Probability analysis with multiple pieces

EDGE CASES:
==========
- **k = 0**: Trivial case, probability = 1
- **n = 1**: Single cell, knight must exit after any move
- **Corner Start**: Minimum survival probability
- **Center Start**: Maximum survival probability for given n

OPTIMIZATION TECHNIQUES:
=======================
**Early Termination**: Stop when all probability has exited
**Symmetry Exploitation**: Reduce computation using board symmetries
**Sparse Representation**: Track only non-zero probability cells
**Precomputation**: Cache results for common (n,k) combinations

This problem elegantly demonstrates how probability theory
and dynamic programming combine to analyze stochastic
processes, showcasing practical applications of Markov
chain analysis in discrete probability calculations.
"""
