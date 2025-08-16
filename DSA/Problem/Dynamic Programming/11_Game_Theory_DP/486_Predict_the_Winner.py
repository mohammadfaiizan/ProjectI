"""
LeetCode 486: Predict the Winner
Difficulty: Medium
Category: Game Theory DP - Minimax Optimization

PROBLEM DESCRIPTION:
===================
You are given an integer array nums. Two players are playing a game with this array: player 1 and player 2.

Player 1 and player 2 take turns, with player 1 starting first. Both players start the game with a score of 0. At each turn, the player takes one of the numbers from either end of the array (i.e., nums[i] or nums[j]) and adds it to their score. The game ends when there are no more elements in the array.

Return true if Player 1 can win this game. If the scores of both players are equal, then player 1 is still the winner.

Example 1:
Input: nums = [1,5,2]
Output: false
Explanation: Initially, player 1 can choose between 1 and 2.
If he chooses 2 (or 1), then player 2 can choose from 1 (or 2) and 5. If player 2 chooses 5, then player 1 will be left with 1 (or 2).
So, final score of player 1 is 1 + 2 = 3, and player 2 is 5.
Hence, player 1 will never be the winner and you need to return false.

Example 2:
Input: nums = [1,5,233,7]
Output: true
Explanation: Player 1 first chooses 1. Then player 2 has to choose between 5 and 7. No matter which number player 2 choose, player 1 can choose 233.
Finally, player 1 has more score (234) than player 2 (12), so you need to return true representing player1 can win.

Constraints:
- 1 <= nums.length <= 20
- 0 <= nums[i] <= 10^7
"""


def predict_the_winner_recursive(nums):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to simulate optimal play by both players.
    
    Time Complexity: O(2^n) - exponential without memoization
    Space Complexity: O(n) - recursion stack
    """
    def max_diff(left, right, is_player1_turn):
        """
        Return the maximum score difference (player1 - player2) 
        that the current player can achieve.
        """
        if left > right:
            return 0
        
        if is_player1_turn:
            # Player 1 wants to maximize the difference
            take_left = nums[left] + max_diff(left + 1, right, False)
            take_right = nums[right] + max_diff(left, right - 1, False)
            return max(take_left, take_right)
        else:
            # Player 2 wants to minimize the difference (from player 1's perspective)
            take_left = -nums[left] + max_diff(left + 1, right, True)
            take_right = -nums[right] + max_diff(left, right - 1, True)
            return min(take_left, take_right)
    
    return max_diff(0, len(nums) - 1, True) >= 0


def predict_the_winner_memoization(nums):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing subproblems.
    
    Time Complexity: O(n^2) - each subproblem solved once
    Space Complexity: O(n^2) - memoization table
    """
    memo = {}
    
    def max_diff(left, right):
        """
        Return max score difference (current player - opponent)
        that current player can achieve on nums[left:right+1]
        """
        if left > right:
            return 0
        
        if (left, right) in memo:
            return memo[(left, right)]
        
        # Current player can take from left or right
        # Opponent will play optimally on remaining subarray
        take_left = nums[left] - max_diff(left + 1, right)
        take_right = nums[right] - max_diff(left, right - 1)
        
        result = max(take_left, take_right)
        memo[(left, right)] = result
        return result
    
    return max_diff(0, len(nums) - 1) >= 0


def predict_the_winner_dp(nums):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Use bottom-up DP to solve the game optimally.
    
    Time Complexity: O(n^2) - fill DP table
    Space Complexity: O(n^2) - DP table
    """
    n = len(nums)
    
    # dp[i][j] = max score difference (current player - opponent)
    # that current player can achieve on nums[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single element
    for i in range(n):
        dp[i][i] = nums[i]
    
    # Fill DP table for increasing subarray lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Current player takes nums[i], opponent plays optimally on nums[i+1:j+1]
            take_left = nums[i] - dp[i + 1][j]
            
            # Current player takes nums[j], opponent plays optimally on nums[i:j]
            take_right = nums[j] - dp[i][j - 1]
            
            dp[i][j] = max(take_left, take_right)
    
    return dp[0][n - 1] >= 0


def predict_the_winner_optimized_space(nums):
    """
    SPACE-OPTIMIZED DP:
    ==================
    Use 1D array since we only need previous row.
    
    Time Complexity: O(n^2) - same as 2D DP
    Space Complexity: O(n) - optimized space
    """
    n = len(nums)
    dp = nums[:]  # Initialize with base case
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            take_left = nums[i] - dp[i + 1]
            take_right = nums[j] - dp[i]
            
            dp[i] = max(take_left, take_right)
    
    return dp[0] >= 0


def predict_the_winner_with_analysis(nums):
    """
    GAME THEORY DP WITH DETAILED ANALYSIS:
    =====================================
    Solve the game and provide detailed strategic insights.
    
    Time Complexity: O(n^2) - standard DP
    Space Complexity: O(n^2) - DP table + analysis
    """
    n = len(nums)
    
    analysis = {
        'nums': nums[:],
        'total_sum': sum(nums),
        'game_tree_size': 2 ** n,
        'optimal_moves': [],
        'score_analysis': {},
        'strategy_insights': []
    }
    
    # DP with move tracking
    dp = [[0] * n for _ in range(n)]
    moves = [[None] * n for _ in range(n)]
    
    # Base case
    for i in range(n):
        dp[i][i] = nums[i]
        moves[i][i] = ('take', i)
    
    # Fill DP table with move tracking
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            take_left = nums[i] - dp[i + 1][j]
            take_right = nums[j] - dp[i][j - 1]
            
            if take_left >= take_right:
                dp[i][j] = take_left
                moves[i][j] = ('left', i)
            else:
                dp[i][j] = take_right
                moves[i][j] = ('right', j)
    
    # Reconstruct optimal game
    def reconstruct_game():
        game_sequence = []
        i, j = 0, n - 1
        is_player1_turn = True
        
        while i <= j:
            if moves[i][j] is None:
                break
                
            move_type, pos = moves[i][j]
            
            if move_type == 'left':
                game_sequence.append({
                    'player': 1 if is_player1_turn else 2,
                    'action': 'take_left',
                    'position': i,
                    'value': nums[i],
                    'remaining': nums[i+1:j+1] if i+1 <= j else []
                })
                i += 1
            else:  # take_right
                game_sequence.append({
                    'player': 1 if is_player1_turn else 2,
                    'action': 'take_right',
                    'position': j,
                    'value': nums[j],
                    'remaining': nums[i:j] if i <= j-1 else []
                })
                j -= 1
            
            is_player1_turn = not is_player1_turn
        
        return game_sequence
    
    game_sequence = reconstruct_game()
    analysis['optimal_moves'] = game_sequence
    
    # Calculate final scores
    player1_score = sum(move['value'] for move in game_sequence if move['player'] == 1)
    player2_score = sum(move['value'] for move in game_sequence if move['player'] == 2)
    
    analysis['score_analysis'] = {
        'player1_score': player1_score,
        'player2_score': player2_score,
        'score_difference': player1_score - player2_score,
        'player1_percentage': player1_score / analysis['total_sum'] * 100,
        'player2_percentage': player2_score / analysis['total_sum'] * 100
    }
    
    # Strategic insights
    if player1_score >= player2_score:
        analysis['winner'] = 'Player 1'
        analysis['strategy_insights'].append("Player 1 has a winning strategy")
    else:
        analysis['winner'] = 'Player 2'
        analysis['strategy_insights'].append("Player 2 has a winning strategy")
    
    # Analyze array properties
    if n % 2 == 0:
        analysis['strategy_insights'].append("Even number of elements - both players get equal number of moves")
    else:
        analysis['strategy_insights'].append("Odd number of elements - Player 1 gets one extra move")
    
    # Check for obvious strategies
    if nums[0] + nums[-1] > sum(nums[1:-1]):
        analysis['strategy_insights'].append("Corner elements dominate - aggressive strategy beneficial")
    
    return dp[0][n - 1] >= 0, analysis


def predict_the_winner_analysis(nums):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the game with detailed strategic insights.
    """
    print(f"Predict the Winner Analysis:")
    print(f"Array: {nums}")
    print(f"Array length: {len(nums)}")
    print(f"Total sum: {sum(nums)}")
    print(f"Average value: {sum(nums) / len(nums):.2f}")
    
    # Different approaches
    recursive = predict_the_winner_recursive(nums)
    memoization = predict_the_winner_memoization(nums)
    dp = predict_the_winner_dp(nums)
    optimized = predict_the_winner_optimized_space(nums)
    
    print(f"Recursive result: {recursive}")
    print(f"Memoization result: {memoization}")
    print(f"DP result: {dp}")
    print(f"Optimized result: {optimized}")
    
    # Detailed analysis
    can_win, analysis = predict_the_winner_with_analysis(nums)
    
    print(f"\nDetailed Game Analysis:")
    print(f"Can Player 1 win: {can_win}")
    print(f"Winner: {analysis['winner']}")
    
    scores = analysis['score_analysis']
    print(f"\nFinal Scores:")
    print(f"Player 1: {scores['player1_score']} ({scores['player1_percentage']:.1f}%)")
    print(f"Player 2: {scores['player2_score']} ({scores['player2_percentage']:.1f}%)")
    print(f"Score difference: {scores['score_difference']}")
    
    print(f"\nOptimal Game Sequence:")
    for i, move in enumerate(analysis['optimal_moves']):
        print(f"  Move {i+1}: Player {move['player']} takes {move['value']} from {move['action']}")
        if move['remaining']:
            print(f"           Remaining: {move['remaining']}")
    
    print(f"\nStrategy Insights:")
    for insight in analysis['strategy_insights']:
        print(f"  • {insight}")
    
    return can_win


def predict_the_winner_variants():
    """
    GAME THEORY VARIANTS:
    ====================
    Different game scenarios and modifications.
    """
    
    def predict_winner_k_moves(nums, k):
        """Predict winner when each player can take up to k elements per turn"""
        # This is much more complex - simplified version
        if k >= len(nums):
            return sum(nums) >= 0  # First player takes everything
        
        # Use minimax with k moves per turn
        memo = {}
        
        def minimax(left, right, is_player1_turn, moves_left):
            if left > right:
                return 0
            
            state = (left, right, is_player1_turn, moves_left)
            if state in memo:
                return memo[state]
            
            if moves_left == 0:
                # Switch turns
                result = minimax(left, right, not is_player1_turn, k)
            else:
                if is_player1_turn:
                    # Player 1 maximizes
                    take_left = nums[left] + minimax(left + 1, right, True, moves_left - 1)
                    take_right = nums[right] + minimax(left, right - 1, True, moves_left - 1)
                    switch_turn = minimax(left, right, False, k)
                    result = max(take_left, take_right, switch_turn)
                else:
                    # Player 2 minimizes (from player 1's perspective)
                    take_left = -nums[left] + minimax(left + 1, right, False, moves_left - 1)
                    take_right = -nums[right] + minimax(left, right - 1, False, moves_left - 1)
                    switch_turn = minimax(left, right, True, k)
                    result = min(take_left, take_right, switch_turn)
            
            memo[state] = result
            return result
        
        return minimax(0, len(nums) - 1, True, k) >= 0
    
    def predict_winner_with_skips(nums, skips_allowed):
        """Predict winner when players can skip turns"""
        memo = {}
        
        def minimax(left, right, is_player1_turn, p1_skips, p2_skips):
            if left > right:
                return 0
            
            state = (left, right, is_player1_turn, p1_skips, p2_skips)
            if state in memo:
                return memo[state]
            
            results = []
            
            if is_player1_turn:
                # Player 1's turn - maximize
                # Take left
                results.append(nums[left] + minimax(left + 1, right, False, p1_skips, p2_skips))
                # Take right
                results.append(nums[right] + minimax(left, right - 1, False, p1_skips, p2_skips))
                # Skip (if allowed)
                if p1_skips > 0:
                    results.append(minimax(left, right, False, p1_skips - 1, p2_skips))
                
                result = max(results)
            else:
                # Player 2's turn - minimize (from player 1's perspective)
                # Take left
                results.append(-nums[left] + minimax(left + 1, right, True, p1_skips, p2_skips))
                # Take right
                results.append(-nums[right] + minimax(left, right - 1, True, p1_skips, p2_skips))
                # Skip (if allowed)
                if p2_skips > 0:
                    results.append(minimax(left, right, True, p1_skips, p2_skips - 1))
                
                result = min(results)
            
            memo[state] = result
            return result
        
        return minimax(0, len(nums) - 1, True, skips_allowed, skips_allowed) >= 0
    
    def count_winning_positions(nums):
        """Count how many starting positions lead to player 1 victory"""
        n = len(nums)
        winning_count = 0
        
        for start in range(n):
            # Rotate array to start from different position
            rotated = nums[start:] + nums[:start]
            if predict_the_winner_dp(rotated):
                winning_count += 1
        
        return winning_count
    
    # Test variants
    test_arrays = [
        [1, 5, 2],
        [1, 5, 233, 7],
        [1, 3, 7, 9],
        [20, 30, 2, 2, 2, 10]
    ]
    
    print("Game Theory Variants:")
    print("=" * 50)
    
    for nums in test_arrays:
        print(f"\nArray: {nums}")
        
        basic_result = predict_the_winner_dp(nums)
        print(f"Basic game: Player 1 can win: {basic_result}")
        
        # K-moves variant
        k_moves_result = predict_winner_k_moves(nums, 2)
        print(f"With 2 moves per turn: Player 1 can win: {k_moves_result}")
        
        # Skips variant
        if len(nums) <= 6:  # Only for small arrays due to complexity
            skip_result = predict_winner_with_skips(nums, 1)
            print(f"With 1 skip allowed: Player 1 can win: {skip_result}")
        
        # Winning positions
        winning_positions = count_winning_positions(nums)
        print(f"Winning starting positions: {winning_positions}/{len(nums)}")


# Test cases
def test_predict_the_winner():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 5, 2], False),
        ([1, 5, 233, 7], True),
        ([1], True),
        ([1, 2], True),
        ([1, 3, 1], False),
        ([3, 7, 2, 3], True),
        ([1, 2, 3, 4, 5], False),
        ([5, 4, 3, 2, 1], True)
    ]
    
    print("Testing Predict the Winner Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"nums = {nums}")
        print(f"Expected: {expected}")
        
        recursive = predict_the_winner_recursive(nums)
        memoization = predict_the_winner_memoization(nums)
        dp = predict_the_winner_dp(nums)
        optimized = predict_the_winner_optimized_space(nums)
        
        print(f"Recursive:        {recursive} {'✓' if recursive == expected else '✗'}")
        print(f"Memoization:      {memoization} {'✓' if memoization == expected else '✗'}")
        print(f"DP:               {dp} {'✓' if dp == expected else '✗'}")
        print(f"Optimized:        {optimized} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    predict_the_winner_analysis([1, 5, 233, 7])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    predict_the_winner_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MINIMAX PRINCIPLE: Each player plays optimally to maximize their advantage")
    print("2. ZERO-SUM GAME: One player's gain equals the other's loss")
    print("3. OPTIMAL SUBSTRUCTURE: Optimal strategy built from optimal sub-strategies")
    print("4. GAME TREE PRUNING: DP eliminates redundant game state exploration")
    print("5. SCORE DIFFERENCE: Focus on relative advantage rather than absolute scores")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Game AI: Optimal strategy computation for two-player games")
    print("• Economics: Competitive market analysis and bidding strategies")
    print("• Resource Allocation: Competitive resource distribution")
    print("• Algorithm Design: Minimax and game theory applications")
    print("• Decision Theory: Optimal decision making under competition")


if __name__ == "__main__":
    test_predict_the_winner()


"""
PREDICT THE WINNER - FUNDAMENTAL GAME THEORY DP:
================================================

This problem establishes core Game Theory DP principles:
- Minimax optimization with optimal play assumption
- Zero-sum game analysis with score difference focus
- Interval DP on game states with player alternation
- Strategic decision making under perfect information

KEY INSIGHTS:
============
1. **MINIMAX PRINCIPLE**: Each player plays optimally to maximize their own advantage
2. **ZERO-SUM GAME**: One player's gain exactly equals the other player's loss
3. **OPTIMAL SUBSTRUCTURE**: Optimal strategy for current state built from optimal sub-strategies
4. **SCORE DIFFERENCE**: Focus on relative advantage (player1 - player2) rather than absolute scores
5. **PERFECT INFORMATION**: Both players have complete knowledge of game state

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(2^n) time, O(n) space
   - Direct minimax simulation with exponential complexity
   - Simple but inefficient without memoization

2. **Memoization**: O(n²) time, O(n²) space
   - Top-down DP with subproblem caching
   - Eliminates redundant recursive calls

3. **Bottom-up DP**: O(n²) time, O(n²) space
   - Iterative DP building solutions from smaller subproblems
   - Most common approach for interval DP

4. **Space-Optimized**: O(n²) time, O(n) space
   - 1D array optimization using DP dependency pattern
   - Optimal space complexity for this problem

CORE MINIMAX ALGORITHM:
======================
```python
def predictTheWinner(nums):
    n = len(nums)
    # dp[i][j] = max score difference (current player - opponent)
    # that current player can achieve on nums[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single element
    for i in range(n):
        dp[i][i] = nums[i]
    
    # Fill for increasing interval lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Take left: gain nums[i], opponent plays optimally on rest
            take_left = nums[i] - dp[i + 1][j]
            
            # Take right: gain nums[j], opponent plays optimally on rest
            take_right = nums[j] - dp[i][j - 1]
            
            dp[i][j] = max(take_left, take_right)
    
    return dp[0][n - 1] >= 0
```

GAME THEORY FORMULATION:
=======================
**State Definition**: `dp[i][j]` = maximum score advantage current player can achieve on subarray nums[i:j+1]

**State Transition**: 
- Take left: `nums[i] - dp[i+1][j]` (gain nums[i], face optimal opponent play)
- Take right: `nums[j] - dp[i][j-1]` (gain nums[j], face optimal opponent play)
- Choose maximum of these options

**Minimax Logic**: Current player maximizes advantage, knowing opponent will minimize it in subsequent turns

SCORE DIFFERENCE ANALYSIS:
=========================
**Why Score Difference**: Simplifies two-player optimization to single-player maximization

**Interpretation**: `dp[i][j] ≥ 0` means current player can achieve non-negative advantage

**Advantage**: Converts complex two-score tracking to single value optimization

OPTIMAL PLAY ASSUMPTION:
=======================
**Perfect Rationality**: Both players always make optimal moves
**Complete Information**: All game state visible to both players
**Deterministic**: No randomness in game mechanics
**Sequential**: Players alternate turns with clear ordering

INTERVAL DP PATTERN:
===================
**Bottom-up Construction**: Build solutions for larger intervals from smaller ones
**Subproblem Overlap**: Many intervals share common sub-intervals
**Optimal Substructure**: Optimal play on interval depends on optimal play on sub-intervals

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n²)
- n² possible intervals [i,j]
- Constant time computation per interval

**Space Complexity**: O(n²) for 2D DP, O(n) for optimized version
- Can be optimized due to DP dependency pattern

**Practical Performance**: Efficient for reasonable array sizes (n ≤ 1000)

STRATEGIC INSIGHTS:
==================
**Even vs Odd Length**: 
- Even length: Both players get equal number of moves
- Odd length: First player gets one extra move (slight advantage)

**Greedy vs Optimal**: 
- Greedy (always take maximum) often suboptimal
- Must consider opponent's response to current move

**Endgame Analysis**: Final moves become forced choices

GAME TREE EXPLORATION:
======================
**Minimax Tree**: Game tree with alternating max/min levels
**Alpha-Beta Pruning**: Not directly applicable due to DP structure
**Move Ordering**: Left/right choice order doesn't affect optimality

APPLICATIONS:
============
- **Game AI**: Optimal strategy computation for turn-based games
- **Economic Games**: Competitive resource allocation and bidding
- **Decision Theory**: Strategic decision making under competition
- **Algorithm Design**: Minimax optimization in various contexts
- **Competitive Analysis**: Market competition and strategy analysis

RELATED PROBLEMS:
================
- **Stone Games**: Various stone-taking game variants
- **Coin Games**: Similar mechanics with different constraints
- **Nim Games**: Combinatorial game theory applications
- **Chess Endgames**: Minimax in complex game trees

VARIANTS:
========
- **K-Move Games**: Players can take multiple elements per turn
- **Skip Options**: Players can choose to skip turns
- **Weighted Scoring**: Different scoring mechanisms
- **Multi-Player**: Extension to more than two players

EDGE CASES:
==========
- **Single Element**: Trivial win for first player
- **Two Elements**: First player chooses maximum
- **All Equal**: First player wins due to equal/tie rule
- **Alternating Pattern**: Various strategic implications

OPTIMIZATION TECHNIQUES:
=======================
**Space Optimization**: Reduce 2D DP to 1D using dependency analysis
**Symmetry**: Exploit symmetric game positions
**Precomputation**: Cache common subproblem results
**Pruning**: Early termination when outcome determined

This problem provides the foundation for understanding
game theory applications in dynamic programming,
demonstrating how competitive scenarios can be
optimally solved through minimax principles and
systematic state space exploration.
"""
