"""
LeetCode 1406: Stone Game III
Difficulty: Hard
Category: Game Theory DP - Multi-Choice Minimax

PROBLEM DESCRIPTION:
===================
Alice and Bob continue their games with piles of stones. There are several stones arranged in a row, and each stone has an associated value which is an integer given in the array stoneValue.

Alice and Bob take turns, with Alice starting first. On each player's turn, that player can take 1, 2, or 3 stones from the first remaining stones in the row.

The score of each player is the sum of the values of the stones taken. The score of each player is initially 0.

The objective of the game is to end with the highest score, and the winner is the player with the highest score. If the scores are tied, then the game results in a draw.

Assuming both players play optimally, determine the result of the game:
- Return "Alice" if Alice can win this game.
- Return "Bob" if Bob can win this game.
- Return "Tie" if the game results in a draw.

Example 1:
Input: stoneValue = [1,2,3,7]
Output: "Bob"
Explanation: 
Alice will always lose. Her best move will be to take three stones, resulting in a score of 6.
Now the row becomes [7] and Bob takes 7.
Bob Score = 7, Alice Score = 6 so Bob wins.

Example 2:
Input: stoneValue = [1,2,3,-9]
Output: "Alice"
Explanation:
Alice must choose all the stones in the first turn to win, since if she chooses 1 or 2 stones, Bob will be able to choose from multiple stones and will have a score higher than hers.

Example 3:
Input: stoneValue = [1,2,3,6]
Output: "Tie"

Constraints:
- 1 <= stoneValue.length <= 50000
- -1000 <= stoneValue[i] <= 1000
"""


def stone_game_iii_recursive(stoneValue):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to explore all possible moves.
    
    Time Complexity: O(3^n) - three choices per position
    Space Complexity: O(n) - recursion stack
    """
    def max_score_diff(start):
        """
        Return max score difference (current player - opponent)
        that current player can achieve from position start
        """
        if start >= len(stoneValue):
            return 0
        
        best = float('-inf')
        current_score = 0
        
        # Try taking 1, 2, or 3 stones
        for take in range(1, min(4, len(stoneValue) - start + 1)):
            current_score += stoneValue[start + take - 1]
            # Current player gets current_score, opponent gets optimal play on rest
            opponent_diff = max_score_diff(start + take)
            total_diff = current_score - opponent_diff
            best = max(best, total_diff)
        
        return best
    
    diff = max_score_diff(0)
    
    if diff > 0:
        return "Alice"
    elif diff < 0:
        return "Bob"
    else:
        return "Tie"


def stone_game_iii_memoization(stoneValue):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed states.
    
    Time Complexity: O(n) - each position computed once
    Space Complexity: O(n) - memoization table
    """
    memo = {}
    
    def max_score_diff(start):
        if start >= len(stoneValue):
            return 0
        
        if start in memo:
            return memo[start]
        
        best = float('-inf')
        current_score = 0
        
        for take in range(1, min(4, len(stoneValue) - start + 1)):
            current_score += stoneValue[start + take - 1]
            opponent_diff = max_score_diff(start + take)
            total_diff = current_score - opponent_diff
            best = max(best, total_diff)
        
        memo[start] = best
        return best
    
    diff = max_score_diff(0)
    
    if diff > 0:
        return "Alice"
    elif diff < 0:
        return "Bob"
    else:
        return "Tie"


def stone_game_iii_dp_bottom_up(stoneValue):
    """
    BOTTOM-UP DP APPROACH:
    =====================
    Use iterative DP from end to beginning.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - DP array
    """
    n = len(stoneValue)
    
    # dp[i] = max score difference current player can achieve from position i
    dp = [0] * (n + 3)  # Extra space to avoid boundary checks
    
    # Fill DP table from end to beginning
    for i in range(n - 1, -1, -1):
        dp[i] = float('-inf')
        current_score = 0
        
        # Try taking 1, 2, or 3 stones
        for take in range(1, min(4, n - i + 1)):
            current_score += stoneValue[i + take - 1]
            # Current player gets current_score, opponent gets optimal on rest
            total_diff = current_score - dp[i + take]
            dp[i] = max(dp[i], total_diff)
    
    if dp[0] > 0:
        return "Alice"
    elif dp[0] < 0:
        return "Bob"
    else:
        return "Tie"


def stone_game_iii_space_optimized(stoneValue):
    """
    SPACE-OPTIMIZED DP:
    ==================
    Use only the last 3 DP values since we only look ahead 3 positions.
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    """
    n = len(stoneValue)
    
    # Only need last 3 values: dp[i+1], dp[i+2], dp[i+3]
    dp = [0, 0, 0]  # dp[i+1], dp[i+2], dp[i+3]
    
    for i in range(n - 1, -1, -1):
        current_best = float('-inf')
        current_score = 0
        
        for take in range(1, min(4, n - i + 1)):
            current_score += stoneValue[i + take - 1]
            # dp[take-1] represents dp[i+take] from the perspective of position i
            opponent_score = dp[take - 1] if take <= 3 else 0
            current_best = max(current_best, current_score - opponent_score)
        
        # Shift the DP values
        dp = [current_best, dp[0], dp[1]]
    
    if dp[0] > 0:
        return "Alice"
    elif dp[0] < 0:
        return "Bob"
    else:
        return "Tie"


def stone_game_iii_with_analysis(stoneValue):
    """
    STONE GAME III WITH DETAILED ANALYSIS:
    =====================================
    Solve the game and provide comprehensive strategic insights.
    
    Time Complexity: O(n) - DP computation + analysis
    Space Complexity: O(n) - DP table + analysis data
    """
    n = len(stoneValue)
    
    analysis = {
        'stone_values': stoneValue[:],
        'num_stones': n,
        'total_sum': sum(stoneValue),
        'positive_stones': sum(v for v in stoneValue if v > 0),
        'negative_stones': sum(v for v in stoneValue if v < 0),
        'optimal_moves': [],
        'score_analysis': {},
        'strategy_insights': []
    }
    
    # DP with move tracking
    dp = [0] * (n + 3)
    best_moves = [0] * n  # Track optimal number of stones to take at each position
    
    # Fill DP table and track optimal moves
    for i in range(n - 1, -1, -1):
        dp[i] = float('-inf')
        current_score = 0
        best_take = 1
        
        for take in range(1, min(4, n - i + 1)):
            current_score += stoneValue[i + take - 1]
            total_diff = current_score - dp[i + take]
            
            if total_diff > dp[i]:
                dp[i] = total_diff
                best_take = take
        
        best_moves[i] = best_take
    
    # Reconstruct optimal game
    def reconstruct_game():
        moves = []
        pos = 0
        is_alice_turn = True
        
        while pos < n:
            take = best_moves[pos]
            taken_stones = stoneValue[pos:pos + take]
            score = sum(taken_stones)
            
            moves.append({
                'player': 'Alice' if is_alice_turn else 'Bob',
                'position': pos,
                'stones_taken': take,
                'stone_values': taken_stones,
                'score': score,
                'remaining': stoneValue[pos + take:] if pos + take < n else []
            })
            
            pos += take
            is_alice_turn = not is_alice_turn
        
        return moves
    
    game_moves = reconstruct_game()
    analysis['optimal_moves'] = game_moves
    
    # Calculate final scores
    alice_score = sum(move['score'] for move in game_moves if move['player'] == 'Alice')
    bob_score = sum(move['score'] for move in game_moves if move['player'] == 'Bob')
    
    analysis['score_analysis'] = {
        'alice_score': alice_score,
        'bob_score': bob_score,
        'score_difference': alice_score - bob_score,
        'optimal_diff': dp[0],
        'verification': alice_score + bob_score == analysis['total_sum']
    }
    
    # Determine winner
    if dp[0] > 0:
        winner = "Alice"
    elif dp[0] < 0:
        winner = "Bob"
    else:
        winner = "Tie"
    
    analysis['winner'] = winner
    
    # Strategic insights
    analysis['strategy_insights'].append(f"Game result: {winner}")
    analysis['strategy_insights'].append(f"Alice's advantage: {dp[0]}")
    
    if analysis['negative_stones'] < 0:
        analysis['strategy_insights'].append("Negative stones present - avoiding them is crucial")
    
    if analysis['positive_stones'] > abs(analysis['negative_stones']):
        analysis['strategy_insights'].append("More positive than negative value - aggressive play beneficial")
    
    # Analyze move patterns
    take_patterns = [move['stones_taken'] for move in game_moves]
    if take_patterns:
        analysis['strategy_insights'].append(f"Optimal move pattern: {take_patterns}")
        
        avg_take = sum(take_patterns) / len(take_patterns)
        analysis['strategy_insights'].append(f"Average stones taken per move: {avg_take:.1f}")
    
    return winner, analysis


def stone_game_iii_analysis(stoneValue):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze Stone Game III with detailed strategic insights.
    """
    print(f"Stone Game III Analysis:")
    print(f"Stone values: {stoneValue}")
    print(f"Number of stones: {len(stoneValue)}")
    print(f"Total sum: {sum(stoneValue)}")
    
    positive_sum = sum(v for v in stoneValue if v > 0)
    negative_sum = sum(v for v in stoneValue if v < 0)
    print(f"Positive stones sum: {positive_sum}")
    print(f"Negative stones sum: {negative_sum}")
    print(f"Net positive: {positive_sum + negative_sum}")
    
    # Different approaches
    if len(stoneValue) <= 20:
        try:
            recursive = stone_game_iii_recursive(stoneValue)
            print(f"Recursive result: {recursive}")
        except:
            print("Recursive: Too slow")
    
    memoization = stone_game_iii_memoization(stoneValue)
    bottom_up = stone_game_iii_dp_bottom_up(stoneValue)
    optimized = stone_game_iii_space_optimized(stoneValue)
    
    print(f"Memoization result: {memoization}")
    print(f"Bottom-up DP result: {bottom_up}")
    print(f"Space-optimized result: {optimized}")
    
    # Detailed analysis
    detailed_result, analysis = stone_game_iii_with_analysis(stoneValue)
    
    print(f"\nDetailed Game Analysis:")
    print(f"Winner: {detailed_result}")
    print(f"Alice's optimal advantage: {analysis['score_analysis']['optimal_diff']}")
    
    scores = analysis['score_analysis']
    print(f"\nFinal Scores:")
    print(f"Alice: {scores['alice_score']}")
    print(f"Bob: {scores['bob_score']}")
    print(f"Difference (Alice - Bob): {scores['score_difference']}")
    print(f"Score sum verification: {scores['verification']}")
    
    print(f"\nOptimal Game Sequence:")
    for i, move in enumerate(analysis['optimal_moves']):
        print(f"  Move {i+1}: {move['player']} takes {move['stones_taken']} stones "
              f"{move['stone_values']} = {move['score']} points")
        if move['remaining']:
            print(f"           Remaining: {move['remaining']}")
    
    print(f"\nStrategy Insights:")
    for insight in analysis['strategy_insights']:
        print(f"  • {insight}")
    
    # Additional analysis
    print(f"\nGame Properties:")
    print(f"  • Each player can take 1, 2, or 3 stones per turn")
    print(f"  • Stone values can be positive or negative")
    print(f"  • Optimal play considers both gaining points and denying opponent")
    print(f"  • Negative stones create defensive strategic considerations")
    
    return detailed_result


def stone_game_iii_variants():
    """
    STONE GAME III VARIANTS:
    =======================
    Different game rule modifications.
    """
    
    def stone_game_iii_variable_choices(stoneValue, max_take):
        """Stone Game III with variable maximum stones per turn"""
        n = len(stoneValue)
        dp = [0] * (n + max_take)
        
        for i in range(n - 1, -1, -1):
            dp[i] = float('-inf')
            current_score = 0
            
            for take in range(1, min(max_take + 1, n - i + 1)):
                current_score += stoneValue[i + take - 1]
                total_diff = current_score - dp[i + take]
                dp[i] = max(dp[i], total_diff)
        
        if dp[0] > 0:
            return "Alice"
        elif dp[0] < 0:
            return "Bob"
        else:
            return "Tie"
    
    def stone_game_iii_with_skips(stoneValue, skip_cost):
        """Stone Game III where players can skip turns with a cost"""
        n = len(stoneValue)
        memo = {}
        
        def max_diff(start):
            if start >= n:
                return 0
            
            if start in memo:
                return memo[start]
            
            best = float('-inf')
            
            # Option 1: Take 1, 2, or 3 stones
            current_score = 0
            for take in range(1, min(4, n - start + 1)):
                current_score += stoneValue[start + take - 1]
                total_diff = current_score - max_diff(start + take)
                best = max(best, total_diff)
            
            # Option 2: Skip turn (pay cost, opponent plays)
            if skip_cost is not None:
                skip_diff = -skip_cost - max_diff(start)
                best = max(best, skip_diff)
            
            memo[start] = best
            return best
        
        diff = max_diff(0)
        
        if diff > 0:
            return "Alice"
        elif diff < 0:
            return "Bob"
        else:
            return "Tie"
    
    def stone_game_iii_multiplayer(stoneValue, num_players):
        """Approximate multi-player version"""
        if num_players == 2:
            return stone_game_iii_memoization(stoneValue)
        
        # Simplified multi-player approximation
        total = sum(stoneValue)
        
        # With more players, individual advantage decreases
        # This is a very rough approximation
        if total > 0:
            return "Player 1 advantage decreases with more players"
        else:
            return "Negative total makes first player disadvantage larger"
    
    def stone_game_iii_with_doubling(stoneValue):
        """Stone Game III where taking 3 stones doubles the score"""
        n = len(stoneValue)
        memo = {}
        
        def max_diff(start):
            if start >= n:
                return 0
            
            if start in memo:
                return memo[start]
            
            best = float('-inf')
            current_score = 0
            
            for take in range(1, min(4, n - start + 1)):
                current_score += stoneValue[start + take - 1]
                
                # Double the score if taking 3 stones
                final_score = current_score * 2 if take == 3 else current_score
                
                total_diff = final_score - max_diff(start + take)
                best = max(best, total_diff)
            
            memo[start] = best
            return best
        
        diff = max_diff(0)
        
        if diff > 0:
            return "Alice"
        elif diff < 0:
            return "Bob"
        else:
            return "Tie"
    
    # Test variants
    test_arrays = [
        [1, 2, 3, 7],
        [1, 2, 3, -9],
        [1, 2, 3, 6],
        [-1, -2, -3],
        [20, 30, -50, 40]
    ]
    
    print("Stone Game III Variants:")
    print("=" * 50)
    
    for stoneValue in test_arrays:
        print(f"\nStone values: {stoneValue}")
        
        basic_result = stone_game_iii_memoization(stoneValue)
        print(f"Basic Stone Game III: {basic_result}")
        
        # Variable choices variant
        var_choice_result = stone_game_iii_variable_choices(stoneValue, 4)
        print(f"With max 4 stones per turn: {var_choice_result}")
        
        # With skips variant
        skip_result = stone_game_iii_with_skips(stoneValue, 5)
        print(f"With skip option (cost 5): {skip_result}")
        
        # With doubling variant
        double_result = stone_game_iii_with_doubling(stoneValue)
        print(f"With 3-stone doubling: {double_result}")
        
        # Multi-player variant
        if len(stoneValue) <= 6:
            multi_result = stone_game_iii_multiplayer(stoneValue, 3)
            print(f"Three players: {multi_result}")


# Test cases
def test_stone_game_iii():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 2, 3, 7], "Bob"),
        ([1, 2, 3, -9], "Alice"),
        ([1, 2, 3, 6], "Tie"),
        ([1], "Alice"),
        ([1, 2], "Alice"),
        ([1, 2, 3], "Alice"),
        ([-1, -2, -3], "Alice"),
        ([20, 30, -50, 40], "Alice")
    ]
    
    print("Testing Stone Game III Solutions:")
    print("=" * 70)
    
    for i, (stoneValue, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"stoneValue = {stoneValue}")
        print(f"Expected: {expected}")
        
        # Skip recursive for large cases
        if len(stoneValue) <= 15:
            try:
                recursive = stone_game_iii_recursive(stoneValue)
                print(f"Recursive:        {recursive} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = stone_game_iii_memoization(stoneValue)
        bottom_up = stone_game_iii_dp_bottom_up(stoneValue)
        optimized = stone_game_iii_space_optimized(stoneValue)
        
        print(f"Memoization:      {memoization} {'✓' if memoization == expected else '✗'}")
        print(f"Bottom-up DP:     {bottom_up} {'✓' if bottom_up == expected else '✗'}")
        print(f"Space-optimized:  {optimized} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    stone_game_iii_analysis([1, 2, 3, -9])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    stone_game_iii_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. FIXED CHOICES: Each player can take 1, 2, or 3 stones per turn")
    print("2. NEGATIVE VALUES: Negative stones add defensive strategic layer")
    print("3. LINEAR DP: State space is linear in array length")
    print("4. OPTIMAL SUBSTRUCTURE: Optimal choice depends on optimal opponent play")
    print("5. SPACE OPTIMIZATION: Only need last 3 DP values due to limited lookahead")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Resource Selection: Fixed choice constraints in competitive scenarios")
    print("• Risk Management: Balancing gains and losses in sequential decisions")
    print("• Game Design: Multi-choice turn-based games with value optimization")
    print("• Economic Modeling: Sequential resource allocation with limited options")
    print("• Strategic Planning: Optimal decision making with constrained choices")


if __name__ == "__main__":
    test_stone_game_iii()


"""
STONE GAME III - MULTI-CHOICE MINIMAX WITH VALUE OPTIMIZATION:
==============================================================

This problem demonstrates constrained choice Game Theory DP:
- Fixed choice set (1, 2, or 3 stones) creates predictable branching
- Positive and negative values add defensive strategic considerations
- Linear state space enables efficient space optimization
- Score difference optimization simplifies two-player tracking

KEY INSIGHTS:
============
1. **FIXED CHOICE SET**: Limited options (1-3 stones) create manageable branching factor
2. **NEGATIVE VALUES**: Negative stones introduce defensive play and risk management
3. **LINEAR STATE SPACE**: Only position matters, not previous choices
4. **SCORE DIFFERENCE**: Focus on advantage rather than absolute scores
5. **SPACE OPTIMIZATION**: Limited lookahead (3 positions) enables O(1) space solution

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(3^n) time, O(n) space
   - Three choices per position without memoization
   - Exponential complexity impractical for large inputs

2. **Memoization**: O(n) time, O(n) space
   - Top-down DP with state caching
   - Each position computed once with three choice evaluation

3. **Bottom-up DP**: O(n) time, O(n) space
   - Iterative DP building from end to beginning
   - More intuitive for understanding optimal substructure

4. **Space-Optimized**: O(n) time, O(1) space
   - Only track last 3 DP values due to limited lookahead
   - Optimal space complexity for this specific problem

CORE FIXED-CHOICE MINIMAX ALGORITHM:
===================================
```python
def stoneGameIII(stoneValue):
    n = len(stoneValue)
    # dp[i] = max score difference current player can achieve from position i
    dp = [0] * (n + 3)  # Extra space for boundary handling
    
    # Fill from end to beginning
    for i in range(n - 1, -1, -1):
        dp[i] = float('-inf')
        current_score = 0
        
        # Try taking 1, 2, or 3 stones
        for take in range(1, min(4, n - i + 1)):
            current_score += stoneValue[i + take - 1]
            # Current gets current_score, opponent optimal on remaining
            total_diff = current_score - dp[i + take]
            dp[i] = max(dp[i], total_diff)
    
    if dp[0] > 0:
        return "Alice"
    elif dp[0] < 0:
        return "Bob"
    else:
        return "Tie"
```

NEGATIVE VALUE STRATEGY:
=======================
**Defensive Considerations**: Negative stones require balancing
- Taking negative stones reduces own score
- Forcing opponent to take negatives can be advantageous
- Strategic trade-offs between direct gain and opponent denial

**Value Distribution Analysis**:
```python
positive_sum = sum(v for v in stoneValue if v > 0)
negative_sum = sum(v for v in stoneValue if v < 0)
net_positive = positive_sum + negative_sum
```

SPACE OPTIMIZATION TECHNIQUE:
============================
**Limited Lookahead**: Since each player can take at most 3 stones, 
current decision only depends on next 3 positions

**Rolling Array**: 
```python
# Instead of dp[0...n], use only dp[0], dp[1], dp[2]
dp = [0, 0, 0]  # dp[i+1], dp[i+2], dp[i+3]

for i in range(n-1, -1, -1):
    current_best = compute_optimal(i, dp)
    dp = [current_best, dp[0], dp[1]]  # Shift values
```

CHOICE EVALUATION PATTERN:
=========================
**Fixed Branching**: Always exactly 3 choices to evaluate (if available)
**Cumulative Scoring**: Build score incrementally as more stones taken
```python
current_score = 0
for take in range(1, min(4, remaining + 1)):
    current_score += stoneValue[position + take - 1]
    evaluate_choice(current_score, future_state)
```

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n)
- Each position evaluated once
- Constant work (3 choices) per position

**Space Complexity**: O(n) for basic DP, O(1) for optimized
- Basic: dp array of size n
- Optimized: only 3 values needed

**Practical Performance**: Very efficient, handles large inputs well

STRATEGIC CONSIDERATIONS:
========================
**Aggressive vs Conservative**: 
- Taking 3 stones maximizes immediate control
- Taking 1 stone preserves future options
- Taking 2 stones balances control and flexibility

**Negative Avoidance**: 
- Sometimes taking more stones to avoid future negatives
- Forcing opponent into negative positions

**Endgame Planning**: 
- Final moves often forced by remaining stone count
- Critical to plan approach to endgame positions

OPTIMAL SUBSTRUCTURE:
====================
**Recursive Relation**: `dp[i] = max over take ∈ {1,2,3} of (sum[i:i+take] - dp[i+take])`

**Independence**: Optimal choice at position i independent of how position i was reached

**Boundary Conditions**: When fewer than 3 stones remain, choices are naturally limited

APPLICATIONS:
============
- **Resource Allocation**: Sequential choices with limited options per turn
- **Risk Management**: Balancing positive and negative outcomes
- **Game Design**: Turn-based games with constrained choice sets
- **Decision Theory**: Multi-option sequential optimization
- **Economic Modeling**: Investment decisions with limited alternatives

RELATED PROBLEMS:
================
- **Stone Game I/II**: Different constraint patterns
- **House Robber**: Similar linear DP with choice constraints
- **Best Time to Buy/Sell Stock**: Sequential decision making
- **Jump Game**: Reachability with choice constraints

VARIANTS:
========
- **Variable Choices**: Different maximum stones per turn
- **Choice Costs**: Taking certain amounts has additional costs
- **Multipliers**: Special bonuses for taking specific amounts
- **Skip Options**: Allow skipping turns with penalties

EDGE CASES:
==========
- **All Negative**: Both players try to minimize losses
- **Single Stone**: Trivial choice for first player
- **Alternating Signs**: Complex defensive strategies required
- **Large Negative at End**: Affects entire strategy

OPTIMIZATION TECHNIQUES:
=======================
**Space Optimization**: Rolling array for O(1) space
**Early Termination**: Not applicable due to fixed choice evaluation
**Precomputation**: Suffix sums not beneficial for this choice pattern
**Boundary Handling**: Careful index management for edge positions

This problem demonstrates how constrained choice sets can
simplify game theory analysis while negative values add
strategic depth, requiring balance between aggressive
play and defensive considerations in optimal decision making.
"""
