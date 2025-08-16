"""
LeetCode 1140: Stone Game II
Difficulty: Medium
Category: Game Theory DP - Variable Move Count

PROBLEM DESCRIPTION:
===================
Alice and Bob continue their games with piles of stones. There are a number of piles arranged in a row, and each pile has a positive number of stones piles[i]. The objective of the game is to end with the most stones.

Alice and Bob take turns, with Alice going first. Initially, M = 1.

On each player's turn, that player can take all the stones in the first X remaining piles, where 1 <= X <= 2*M. Then, we set M = max(M, X).

The game ends when there are no more piles left. Assuming both Alice and Bob play optimally, return the maximum number of stones that Alice can collect.

Example 1:
Input: piles = [2,7,9,4,4]
Output: 10
Explanation: If Alice takes one pile at the beginning, Bob takes two piles, then Alice takes 2 piles again. Alice can collect 2 + 4 + 4 = 10 stones.
If Alice takes two piles at the beginning, Bob can take all three remaining piles. In this case, Alice collect 2 + 7 = 9 stones.
So we return 10 since it's larger.

Example 2:
Input: piles = [1,2,3,4,5,100]
Output: 104

Constraints:
- 1 <= piles.length <= 100
- 1 <= piles[i] <= 10^4
"""


def stone_game_ii_recursive(piles):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to explore all possible moves with varying M values.
    
    Time Complexity: O(n^n) - exponential without memoization
    Space Complexity: O(n) - recursion stack
    """
    def max_stones(start, M, is_alice_turn):
        """
        Return max stones the current player can collect
        start: starting pile index
        M: current M value
        is_alice_turn: whether it's Alice's turn
        """
        if start >= len(piles):
            return 0
        
        if is_alice_turn:
            # Alice wants to maximize her stones
            best = 0
            current_sum = 0
            
            # Try taking X piles (1 <= X <= 2*M)
            for X in range(1, min(2 * M + 1, len(piles) - start + 1)):
                current_sum += piles[start + X - 1]
                new_M = max(M, X)
                
                # Alice gets current_sum + her optimal play in remaining game
                alice_future = max_stones(start + X, new_M, False)
                best = max(best, current_sum + alice_future)
            
            return best
        else:
            # Bob wants to minimize Alice's total stones
            best = float('inf')
            
            # Try taking X piles
            for X in range(1, min(2 * M + 1, len(piles) - start + 1)):
                new_M = max(M, X)
                # Alice's stones in remaining game after Bob's move
                alice_future = max_stones(start + X, new_M, True)
                best = min(best, alice_future)
            
            return best if best != float('inf') else 0
    
    return max_stones(0, 1, True)


def stone_game_ii_memoization(piles):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed game states.
    
    Time Complexity: O(n^3) - n positions × n M values × n choices
    Space Complexity: O(n^2) - memoization table
    """
    memo = {}
    
    def max_stones(start, M, is_alice_turn):
        if start >= len(piles):
            return 0
        
        state = (start, M, is_alice_turn)
        if state in memo:
            return memo[state]
        
        if is_alice_turn:
            best = 0
            current_sum = 0
            
            for X in range(1, min(2 * M + 1, len(piles) - start + 1)):
                current_sum += piles[start + X - 1]
                new_M = max(M, X)
                alice_future = max_stones(start + X, new_M, False)
                best = max(best, current_sum + alice_future)
            
            memo[state] = best
            return best
        else:
            best = float('inf')
            
            for X in range(1, min(2 * M + 1, len(piles) - start + 1)):
                new_M = max(M, X)
                alice_future = max_stones(start + X, new_M, True)
                best = min(best, alice_future)
            
            result = best if best != float('inf') else 0
            memo[state] = result
            return result
    
    return max_stones(0, 1, True)


def stone_game_ii_dp_optimized(piles):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Use suffix sums and score difference optimization.
    
    Time Complexity: O(n^3) - optimized constant factors
    Space Complexity: O(n^2) - DP table
    """
    n = len(piles)
    
    # Compute suffix sums for quick range sum calculation
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + piles[i]
    
    # dp[i][M] = max stones current player can get from position i with M value
    memo = {}
    
    def dp(start, M):
        if start >= n:
            return 0
        
        if (start, M) in memo:
            return memo[(start, M)]
        
        # If can take all remaining piles
        if start + 2 * M >= n:
            memo[(start, M)] = suffix_sum[start]
            return suffix_sum[start]
        
        best = 0
        for X in range(1, 2 * M + 1):
            if start + X > n:
                break
            
            # Current player takes X piles, opponent plays optimally on rest
            taken = suffix_sum[start] - suffix_sum[start + X]
            remaining_total = suffix_sum[start + X]
            opponent_best = dp(start + X, max(M, X))
            current_player_from_remaining = remaining_total - opponent_best
            
            total_for_current = taken + current_player_from_remaining
            best = max(best, total_for_current)
        
        memo[(start, M)] = best
        return best
    
    return dp(0, 1)


def stone_game_ii_with_analysis(piles):
    """
    STONE GAME II WITH DETAILED ANALYSIS:
    ====================================
    Solve the game and provide comprehensive strategic insights.
    
    Time Complexity: O(n^3) - DP computation + analysis
    Space Complexity: O(n^2) - DP table + analysis data
    """
    n = len(piles)
    
    analysis = {
        'piles': piles[:],
        'total_stones': sum(piles),
        'num_piles': n,
        'optimal_moves': [],
        'm_evolution': [],
        'score_analysis': {},
        'strategy_insights': []
    }
    
    # Suffix sums for efficient range calculations
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + piles[i]
    
    # DP with move tracking
    memo = {}
    best_moves = {}
    
    def dp(start, M):
        if start >= n:
            return 0
        
        if (start, M) in memo:
            return memo[(start, M)]
        
        if start + 2 * M >= n:
            result = suffix_sum[start]
            memo[(start, M)] = result
            best_moves[(start, M)] = n - start  # Take all remaining
            return result
        
        best_score = 0
        best_X = 1
        
        for X in range(1, 2 * M + 1):
            if start + X > n:
                break
            
            taken = suffix_sum[start] - suffix_sum[start + X]
            remaining_total = suffix_sum[start + X]
            opponent_best = dp(start + X, max(M, X))
            current_from_remaining = remaining_total - opponent_best
            
            total = taken + current_from_remaining
            
            if total > best_score:
                best_score = total
                best_X = X
        
        memo[(start, M)] = best_score
        best_moves[(start, M)] = best_X
        return best_score
    
    alice_optimal = dp(0, 1)
    bob_optimal = analysis['total_stones'] - alice_optimal
    
    # Reconstruct optimal game
    def reconstruct_game():
        moves = []
        start, M = 0, 1
        is_alice_turn = True
        
        while start < n:
            if (start, M) not in best_moves:
                break
            
            X = best_moves[(start, M)]
            taken_piles = piles[start:start + X]
            stones_taken = sum(taken_piles)
            
            moves.append({
                'player': 'Alice' if is_alice_turn else 'Bob',
                'start_position': start,
                'piles_taken': X,
                'piles_indices': list(range(start, start + X)),
                'stones_collected': stones_taken,
                'M_before': M,
                'M_after': max(M, X),
                'remaining_piles': piles[start + X:] if start + X < n else []
            })
            
            start += X
            M = max(M, X)
            is_alice_turn = not is_alice_turn
        
        return moves
    
    game_moves = reconstruct_game()
    analysis['optimal_moves'] = game_moves
    
    # Calculate final scores
    alice_score = sum(move['stones_collected'] for move in game_moves if move['player'] == 'Alice')
    bob_score = sum(move['stones_collected'] for move in game_moves if move['player'] == 'Bob')
    
    analysis['score_analysis'] = {
        'alice_optimal': alice_optimal,
        'alice_actual': alice_score,
        'bob_optimal': bob_optimal,
        'bob_actual': bob_score,
        'total_verification': alice_score + bob_score
    }
    
    # M value evolution
    for move in game_moves:
        analysis['m_evolution'].append({
            'move': f"{move['player']} takes {move['piles_taken']} piles",
            'M_before': move['M_before'],
            'M_after': move['M_after']
        })
    
    # Strategic insights
    analysis['strategy_insights'].append(f"Alice can collect {alice_optimal} out of {analysis['total_stones']} stones")
    analysis['strategy_insights'].append(f"Alice gets {alice_optimal / analysis['total_stones'] * 100:.1f}% of total stones")
    
    if len(game_moves) > 0:
        max_M = max(move['M_after'] for move in game_moves)
        analysis['strategy_insights'].append(f"Maximum M value reached: {max_M}")
        
        first_move = game_moves[0]['piles_taken']
        analysis['strategy_insights'].append(f"Optimal first move: take {first_move} pile(s)")
    
    return alice_optimal, analysis


def stone_game_ii_analysis(piles):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze Stone Game II with detailed strategic insights.
    """
    print(f"Stone Game II Analysis:")
    print(f"Piles: {piles}")
    print(f"Number of piles: {len(piles)}")
    print(f"Total stones: {sum(piles)}")
    print(f"Average pile size: {sum(piles) / len(piles):.2f}")
    
    # Different approaches
    if len(piles) <= 10:
        try:
            recursive = stone_game_ii_recursive(piles)
            print(f"Recursive result: {recursive}")
        except:
            print("Recursive: Too slow")
    
    memoization = stone_game_ii_memoization(piles)
    optimized = stone_game_ii_dp_optimized(piles)
    
    print(f"Memoization result: {memoization}")
    print(f"Optimized DP result: {optimized}")
    
    # Detailed analysis
    detailed_result, analysis = stone_game_ii_with_analysis(piles)
    
    print(f"\nDetailed Game Analysis:")
    print(f"Alice's optimal stones: {detailed_result}")
    print(f"Bob's optimal stones: {analysis['total_stones'] - detailed_result}")
    
    scores = analysis['score_analysis']
    print(f"\nScore Verification:")
    print(f"Alice optimal: {scores['alice_optimal']}, actual: {scores['alice_actual']}")
    print(f"Bob optimal: {scores['bob_optimal']}, actual: {scores['bob_actual']}")
    print(f"Total collected: {scores['total_verification']} / {analysis['total_stones']}")
    
    print(f"\nOptimal Game Sequence:")
    for i, move in enumerate(analysis['optimal_moves']):
        print(f"  Move {i+1}: {move['player']} takes {move['piles_taken']} piles "
              f"(indices {move['piles_indices']}) = {move['stones_collected']} stones")
        print(f"           M: {move['M_before']} → {move['M_after']}")
        if move['remaining_piles']:
            print(f"           Remaining: {move['remaining_piles']}")
    
    print(f"\nM Value Evolution:")
    for evolution in analysis['m_evolution']:
        print(f"  {evolution['move']}: M {evolution['M_before']} → {evolution['M_after']}")
    
    print(f"\nStrategy Insights:")
    for insight in analysis['strategy_insights']:
        print(f"  • {insight}")
    
    return detailed_result


def stone_game_ii_variants():
    """
    STONE GAME II VARIANTS:
    ======================
    Different game rule modifications.
    """
    
    def stone_game_ii_with_minimum_m(piles, min_M):
        """Stone Game II where M never goes below min_M"""
        n = len(piles)
        suffix_sum = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + piles[i]
        
        memo = {}
        
        def dp(start, M):
            if start >= n:
                return 0
            
            effective_M = max(M, min_M)
            
            if (start, effective_M) in memo:
                return memo[(start, effective_M)]
            
            if start + 2 * effective_M >= n:
                result = suffix_sum[start]
                memo[(start, effective_M)] = result
                return result
            
            best = 0
            for X in range(1, 2 * effective_M + 1):
                if start + X > n:
                    break
                
                taken = suffix_sum[start] - suffix_sum[start + X]
                remaining_total = suffix_sum[start + X]
                opponent_best = dp(start + X, max(effective_M, X))
                current_from_remaining = remaining_total - opponent_best
                
                best = max(best, taken + current_from_remaining)
            
            memo[(start, effective_M)] = best
            return best
        
        return dp(0, 1)
    
    def stone_game_ii_with_maximum_m(piles, max_M):
        """Stone Game II where M is capped at max_M"""
        n = len(piles)
        suffix_sum = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + piles[i]
        
        memo = {}
        
        def dp(start, M):
            if start >= n:
                return 0
            
            effective_M = min(M, max_M)
            
            if (start, effective_M) in memo:
                return memo[(start, effective_M)]
            
            if start + 2 * effective_M >= n:
                result = suffix_sum[start]
                memo[(start, effective_M)] = result
                return result
            
            best = 0
            for X in range(1, 2 * effective_M + 1):
                if start + X > n:
                    break
                
                taken = suffix_sum[start] - suffix_sum[start + X]
                remaining_total = suffix_sum[start + X]
                opponent_best = dp(start + X, min(max(effective_M, X), max_M))
                current_from_remaining = remaining_total - opponent_best
                
                best = max(best, taken + current_from_remaining)
            
            memo[(start, effective_M)] = best
            return best
        
        return dp(0, 1)
    
    def stone_game_ii_three_players(piles):
        """Approximate three-player version"""
        # Simplified three-player approximation
        total = sum(piles)
        alice_two_player = stone_game_ii_dp_optimized(piles)
        
        # Rough approximation: with three players, first player advantage decreases
        alice_three_player = alice_two_player * 0.7  # Heuristic adjustment
        
        return alice_three_player
    
    def stone_game_ii_with_cost(piles, take_cost):
        """Stone Game II where taking piles has a cost"""
        n = len(piles)
        suffix_sum = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + piles[i]
        
        memo = {}
        
        def dp(start, M):
            if start >= n:
                return 0
            
            if (start, M) in memo:
                return memo[(start, M)]
            
            if start + 2 * M >= n:
                result = suffix_sum[start] - take_cost  # Cost for taking remaining
                memo[(start, M)] = max(0, result)
                return max(0, result)
            
            best = 0
            for X in range(1, 2 * M + 1):
                if start + X > n:
                    break
                
                taken = suffix_sum[start] - suffix_sum[start + X] - take_cost
                if taken < 0:
                    continue
                
                remaining_total = suffix_sum[start + X]
                opponent_best = dp(start + X, max(M, X))
                current_from_remaining = remaining_total - opponent_best
                
                best = max(best, taken + current_from_remaining)
            
            memo[(start, M)] = best
            return best
        
        return dp(0, 1)
    
    # Test variants
    test_piles = [
        [2, 7, 9, 4, 4],
        [1, 2, 3, 4, 5, 100],
        [10, 20, 30, 40],
        [1, 1, 1, 1, 1, 1]
    ]
    
    print("Stone Game II Variants:")
    print("=" * 50)
    
    for piles in test_piles:
        print(f"\nPiles: {piles}")
        
        basic_result = stone_game_ii_dp_optimized(piles)
        print(f"Basic Stone Game II: {basic_result}")
        
        # Minimum M variant
        min_m_result = stone_game_ii_with_minimum_m(piles, 2)
        print(f"With minimum M=2: {min_m_result}")
        
        # Maximum M variant
        max_m_result = stone_game_ii_with_maximum_m(piles, 3)
        print(f"With maximum M=3: {max_m_result}")
        
        # Three players variant
        three_player_result = stone_game_ii_three_players(piles)
        print(f"Three players (approx): {three_player_result:.1f}")
        
        # With cost variant
        cost_result = stone_game_ii_with_cost(piles, 5)
        print(f"With taking cost 5: {cost_result}")


# Test cases
def test_stone_game_ii():
    """Test all implementations with various inputs"""
    test_cases = [
        ([2, 7, 9, 4, 4], 10),
        ([1, 2, 3, 4, 5, 100], 104),
        ([1], 1),
        ([1, 2], 3),
        ([3, 9, 1, 2], 12),
        ([20, 30, 40, 50], 90),
        ([1, 1, 1, 1, 1, 1], 4)
    ]
    
    print("Testing Stone Game II Solutions:")
    print("=" * 70)
    
    for i, (piles, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"piles = {piles}")
        print(f"Expected: {expected}")
        
        # Skip recursive for large cases
        if len(piles) <= 8:
            try:
                recursive = stone_game_ii_recursive(piles)
                print(f"Recursive:        {recursive:>4} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = stone_game_ii_memoization(piles)
        optimized = stone_game_ii_dp_optimized(piles)
        
        print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        print(f"Optimized:        {optimized:>4} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    stone_game_ii_analysis([2, 7, 9, 4, 4])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    stone_game_ii_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. VARIABLE CONSTRAINTS: M value changes based on previous moves")
    print("2. FORWARD PLANNING: Future M values affect current optimal choices")
    print("3. SUFFIX OPTIMIZATION: Use suffix sums for efficient range calculations")
    print("4. STATE EXPANSION: Larger state space due to varying M parameter")
    print("5. GREEDY INADEQUACY: Local optimization often suboptimal due to M evolution")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Adaptive Game Rules: Games where rules change based on player actions")
    print("• Resource Management: Systems where action capacity grows with usage")
    print("• Strategic Planning: Long-term planning with evolving constraints")
    print("• Economic Modeling: Markets where participant power grows with activity")
    print("• Algorithm Design: Dynamic constraint satisfaction problems")


if __name__ == "__main__":
    test_stone_game_ii()


"""
STONE GAME II - DYNAMIC CONSTRAINT GAME THEORY:
===============================================

This problem introduces dynamic constraints to Game Theory DP:
- Variable move constraints that evolve based on previous choices
- Forward-looking optimization considering future constraint changes
- Complex state space with both position and constraint parameters
- Strategic depth requiring multi-step planning beyond immediate gains

KEY INSIGHTS:
============
1. **VARIABLE CONSTRAINTS**: M value changes dynamically, affecting future move options
2. **FORWARD PLANNING**: Current optimal choice depends on future M values and options
3. **STATE COMPLEXITY**: State space includes both position and current M value
4. **CONSTRAINT EVOLUTION**: M = max(M, X) creates irreversible constraint expansion
5. **STRATEGIC DEPTH**: Simple greedy approaches fail due to constraint interdependencies

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(n^n) time, O(n) space
   - Pure recursive exploration of all game trees
   - Exponential complexity without memoization

2. **Memoization**: O(n³) time, O(n²) space
   - Cache results for (position, M, player) states
   - Dramatic performance improvement over pure recursion

3. **Optimized DP**: O(n³) time, O(n²) space
   - Use suffix sums and score difference optimization
   - Eliminate player parameter through minimax transformation

4. **Strategy Analysis**: O(n³) time, O(n²) space
   - Include optimal move tracking and game reconstruction
   - Essential for understanding strategic dynamics

CORE DYNAMIC CONSTRAINT ALGORITHM:
=================================
```python
def stoneGameII(piles):
    n = len(piles)
    # Suffix sums for efficient range calculations
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + piles[i]
    
    memo = {}
    
    def dp(start, M):
        if start >= n:
            return 0
        
        if (start, M) in memo:
            return memo[(start, M)]
        
        # If can take all remaining piles
        if start + 2 * M >= n:
            memo[(start, M)] = suffix_sum[start]
            return suffix_sum[start]
        
        best = 0
        # Try taking X piles (1 <= X <= 2*M)
        for X in range(1, 2 * M + 1):
            if start + X > n:
                break
            
            # Current player takes X piles
            taken = suffix_sum[start] - suffix_sum[start + X]
            # Opponent's optimal play on remaining
            remaining_total = suffix_sum[start + X]
            opponent_best = dp(start + X, max(M, X))
            current_player_remaining = remaining_total - opponent_best
            
            total = taken + current_player_remaining
            best = max(best, total)
        
        memo[(start, M)] = best
        return best
    
    return dp(0, 1)
```

CONSTRAINT EVOLUTION DYNAMICS:
=============================
**M Update Rule**: `M = max(M, X)` where X is number of piles taken
**Irreversibility**: M can only increase, never decrease
**Strategic Impact**: Taking more piles now enables larger future moves
**Planning Horizon**: Must consider how current M affects future opportunities

SUFFIX SUM OPTIMIZATION:
=======================
**Efficient Range Sums**: Precompute suffix sums for O(1) range calculations
```python
suffix_sum[i] = sum(piles[i:])  # Total stones from position i to end
taken_stones = suffix_sum[start] - suffix_sum[start + X]
```

**Space-Time Tradeoff**: O(n) preprocessing for O(1) range queries

STATE SPACE ANALYSIS:
====================
**State Dimensions**: (position, M_value)
- Position: 0 to n-1 (current pile index)
- M_value: 1 to n (maximum possible M)
- Total states: O(n²)

**State Transitions**: For each valid X ∈ [1, 2M]:
- New position: start + X
- New M: max(M, X)
- Recursive subproblem: opponent's optimal play

MINIMAX TRANSFORMATION:
======================
**Score Difference Approach**: Instead of tracking both player scores separately
- Current player maximizes: own_score - opponent_score
- Opponent minimizes: opponent_score - current_player_score
- Equivalent to: current_player_total = taken + (remaining_total - opponent_optimal)

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n³)
- n² possible (position, M) states
- Up to 2M ≤ 2n choices per state
- Total: O(n² × n) = O(n³)

**Space Complexity**: O(n²)
- Memoization table size
- Suffix sum array: O(n)

**Practical Performance**: Efficient for reasonable pile counts (n ≤ 100)

STRATEGIC PLANNING:
==================
**Multi-Step Lookahead**: Optimal choice considers:
1. Immediate stones gained
2. Future M value impact
3. Opponent's response options
4. Long-term positional advantage

**Constraint Management**: Balance between:
- Taking more stones now
- Preserving future flexibility
- Limiting opponent's options

GAME DYNAMICS:
=============
**Early Game**: Small M limits move options, focus on position
**Mid Game**: M grows, more strategic choices become available
**End Game**: Large M may allow taking all remaining piles

**M Evolution Pattern**:
- Typically increases throughout game
- Aggressive early moves expand future options
- Conservative play may limit strategic flexibility

APPLICATIONS:
============
- **Adaptive Systems**: Rules that evolve based on user behavior
- **Resource Management**: Capacity that grows with utilization
- **Economic Games**: Market power that increases with activity
- **Strategic Planning**: Long-term optimization with evolving constraints
- **Game Design**: Mechanics where player abilities expand over time

RELATED PROBLEMS:
================
- **Stone Game I**: Fixed constraint version (take from ends only)
- **Stone Game III**: Different scoring and move rules
- **Stone Game IV**: Different constraint evolution
- **Adaptive Games**: General category of evolving-rule games

VARIANTS:
========
- **M Decay**: M decreases over time instead of growing
- **M Limits**: Cap on maximum M value
- **Cost Mechanics**: Taking piles has associated costs
- **Multi-Player**: Extension to more than two players

OPTIMIZATION TECHNIQUES:
=======================
**Early Termination**: When start + 2M ≥ n, take all remaining
**Suffix Precomputation**: O(1) range sum calculations
**State Compression**: Eliminate redundant state information
**Pruning**: Skip impossible or dominated moves

This problem demonstrates how dynamic constraints create
rich strategic depth in competitive scenarios, requiring
sophisticated forward planning and constraint management
while maintaining optimal competitive play principles.
"""
