"""
LeetCode 877: Stone Game
Difficulty: Medium
Category: Game Theory DP - Classic Two-Player Game

PROBLEM DESCRIPTION:
===================
Alice and Bob play a game with piles of stones. There are an even number of piles arranged in a row, and each pile has a positive number of stones piles[i].

The objective of the game is to end with the most stones. The total number of stones across all the piles is odd, so there are no ties.

Alice and Bob take turns, with Alice going first. Each turn, a player takes the entire pile of stones either from the beginning or the end of the row. This continues until there are no more piles left, at which point the person with the most stones wins.

Assuming both Alice and Bob play optimally, return true if Alice wins the game, or false if Bob wins.

Example 1:
Input: piles = [5,3,8,4]
Output: true
Explanation: Alice starts first, and can only take the first 5 or the last 4.
Say she takes the first 5, so that the row becomes [3, 8, 4].
If Bob takes 3, then the row becomes [8, 4], and Alice takes 8 to win.
If Bob takes the last 4, then the row becomes [3, 8], and Alice takes 8 to win.
If Alice takes the last 4 first, then the row becomes [5, 3, 8], and Alice can always win.
So Alice wins.

Example 2:
Input: piles = [3,7,2,3]
Output: true

Constraints:
- 2 <= piles.length <= 500
- piles.length is even.
- 1 <= piles[i] <= 500
- The sum of piles[i] is odd.
"""


def stone_game_mathematical_insight(piles):
    """
    MATHEMATICAL INSIGHT:
    ====================
    Alice always wins when there are even number of piles and odd total.
    
    Time Complexity: O(1) - mathematical proof
    Space Complexity: O(1) - no additional space needed
    """
    # With even number of piles and odd total sum, Alice can always win
    # She can choose a strategy to always take from even positions or odd positions
    # One of these strategies will give her more than half the total stones
    return True


def stone_game_dp_classic(piles):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Classic minimax DP solution for educational purposes.
    
    Time Complexity: O(n^2) - fill DP table
    Space Complexity: O(n^2) - DP table
    """
    n = len(piles)
    
    # dp[i][j] = max stones advantage (Alice - Bob) Alice can get from piles[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single pile
    for i in range(n):
        dp[i][i] = piles[i]
    
    # Fill DP table for increasing lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Alice takes pile i, Bob plays optimally on remaining
            take_left = piles[i] - dp[i + 1][j]
            
            # Alice takes pile j, Bob plays optimally on remaining
            take_right = piles[j] - dp[i][j - 1]
            
            dp[i][j] = max(take_left, take_right)
    
    return dp[0][n - 1] > 0


def stone_game_with_scores(piles):
    """
    STONE GAME WITH SCORE TRACKING:
    ===============================
    Calculate actual scores for both players.
    
    Time Complexity: O(n^2) - DP computation
    Space Complexity: O(n^2) - DP table + score tracking
    """
    n = len(piles)
    
    # dp[i][j] = (alice_score, bob_score) for optimal play on piles[i:j+1]
    # starting with Alice's turn
    dp = [[(0, 0)] * n for _ in range(n)]
    
    # Base case: single pile (Alice takes it)
    for i in range(n):
        dp[i][i] = (piles[i], 0)
    
    # Fill DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Alice takes left pile
            alice_left = piles[i]
            bob_score_left, alice_score_left = dp[i + 1][j]  # Bob goes first on remaining
            total_alice_left = alice_left + alice_score_left
            total_bob_left = bob_score_left
            
            # Alice takes right pile
            alice_right = piles[j]
            bob_score_right, alice_score_right = dp[i][j - 1]  # Bob goes first on remaining
            total_alice_right = alice_right + alice_score_right
            total_bob_right = bob_score_right
            
            # Alice chooses the option that maximizes her score
            if total_alice_left - total_bob_left >= total_alice_right - total_bob_right:
                dp[i][j] = (total_alice_left, total_bob_left)
            else:
                dp[i][j] = (total_alice_right, total_bob_right)
    
    alice_score, bob_score = dp[0][n - 1]
    return alice_score > bob_score, (alice_score, bob_score)


def stone_game_with_strategy(piles):
    """
    STONE GAME WITH STRATEGY RECONSTRUCTION:
    =======================================
    Find optimal strategy and reconstruct the game.
    
    Time Complexity: O(n^2) - DP + strategy reconstruction
    Space Complexity: O(n^2) - DP table + strategy tracking
    """
    n = len(piles)
    
    # dp[i][j] = max advantage for current player on piles[i:j+1]
    dp = [[0] * n for _ in range(n)]
    choice = [[None] * n for _ in range(n)]
    
    # Base case
    for i in range(n):
        dp[i][i] = piles[i]
        choice[i][i] = 'single'
    
    # Fill DP table with choice tracking
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            take_left = piles[i] - dp[i + 1][j]
            take_right = piles[j] - dp[i][j - 1]
            
            if take_left >= take_right:
                dp[i][j] = take_left
                choice[i][j] = 'left'
            else:
                dp[i][j] = take_right
                choice[i][j] = 'right'
    
    # Reconstruct optimal strategy
    def reconstruct_game():
        moves = []
        i, j = 0, n - 1
        is_alice_turn = True
        
        while i <= j:
            player = "Alice" if is_alice_turn else "Bob"
            
            if choice[i][j] == 'left':
                moves.append({
                    'player': player,
                    'action': 'take_left',
                    'pile_index': i,
                    'stones': piles[i],
                    'remaining_piles': piles[i+1:j+1] if i+1 <= j else []
                })
                i += 1
            elif choice[i][j] == 'right':
                moves.append({
                    'player': player,
                    'action': 'take_right',
                    'pile_index': j,
                    'stones': piles[j],
                    'remaining_piles': piles[i:j] if i <= j-1 else []
                })
                j -= 1
            else:  # single pile
                moves.append({
                    'player': player,
                    'action': 'take_single',
                    'pile_index': i,
                    'stones': piles[i],
                    'remaining_piles': []
                })
                break
            
            is_alice_turn = not is_alice_turn
        
        return moves
    
    game_moves = reconstruct_game()
    alice_total = sum(move['stones'] for move in game_moves if move['player'] == 'Alice')
    bob_total = sum(move['stones'] for move in game_moves if move['player'] == 'Bob')
    
    return alice_total > bob_total, game_moves, (alice_total, bob_total)


def stone_game_analysis(piles):
    """
    COMPREHENSIVE STONE GAME ANALYSIS:
    =================================
    Analyze the stone game with detailed insights.
    """
    print(f"Stone Game Analysis:")
    print(f"Piles: {piles}")
    print(f"Number of piles: {len(piles)}")
    print(f"Total stones: {sum(piles)}")
    print(f"Average pile size: {sum(piles) / len(piles):.2f}")
    
    # Mathematical insight
    math_result = stone_game_mathematical_insight(piles)
    print(f"Mathematical result: Alice wins: {math_result}")
    
    # DP solution
    dp_result = stone_game_dp_classic(piles)
    print(f"DP result: Alice wins: {dp_result}")
    
    # Score analysis
    score_result, scores = stone_game_with_scores(piles)
    alice_score, bob_score = scores
    print(f"Score analysis: Alice wins: {score_result}")
    print(f"Alice score: {alice_score}, Bob score: {bob_score}")
    
    # Strategy analysis
    strategy_result, moves, final_scores = stone_game_with_strategy(piles)
    alice_final, bob_final = final_scores
    
    print(f"\nOptimal Game Play:")
    for i, move in enumerate(moves):
        print(f"  Move {i+1}: {move['player']} takes {move['stones']} stones from pile {move['pile_index']}")
        if move['remaining_piles']:
            print(f"           Remaining: {move['remaining_piles']}")
    
    print(f"\nFinal Scores: Alice {alice_final}, Bob {bob_final}")
    print(f"Winner: {'Alice' if alice_final > bob_final else 'Bob'}")
    
    # Strategic insights
    print(f"\nStrategic Insights:")
    
    # Analyze pile positions
    even_sum = sum(piles[i] for i in range(0, len(piles), 2))
    odd_sum = sum(piles[i] for i in range(1, len(piles), 2))
    print(f"Even position sum: {even_sum}")
    print(f"Odd position sum: {odd_sum}")
    
    if even_sum > odd_sum:
        print(f"Even positions have more stones - Alice should focus on even positions")
    else:
        print(f"Odd positions have more stones - Alice should focus on odd positions")
    
    # Analyze game properties
    if len(piles) % 2 == 0:
        print(f"Even number of piles - both players get equal number of turns")
    
    if sum(piles) % 2 == 1:
        print(f"Odd total stones - no ties possible")
    
    return math_result


def stone_game_variants():
    """
    STONE GAME VARIANTS:
    ===================
    Different stone game scenarios and modifications.
    """
    
    def stone_game_with_multipliers(piles, multipliers):
        """Stone game where each pile has a score multiplier"""
        n = len(piles)
        
        # Calculate effective values
        values = [piles[i] * multipliers[i] for i in range(n)]
        
        # Use standard stone game DP on effective values
        dp = [[0] * n for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = values[i]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                take_left = values[i] - dp[i + 1][j]
                take_right = values[j] - dp[i][j - 1]
                dp[i][j] = max(take_left, take_right)
        
        return dp[0][n - 1] > 0
    
    def stone_game_three_players(piles):
        """Stone game with three players (simplified)"""
        # This is much more complex - simplified approximation
        n = len(piles)
        if n % 3 != 0:
            return "No fair distribution possible"
        
        # Rough approximation: divide into thirds
        third = n // 3
        player1_sum = sum(piles[:third])
        player2_sum = sum(piles[third:2*third])
        player3_sum = sum(piles[2*third:])
        
        scores = [player1_sum, player2_sum, player3_sum]
        winner = max(range(3), key=lambda i: scores[i])
        
        return f"Player {winner + 1} wins with {scores[winner]} stones"
    
    def stone_game_adjacent_only(piles):
        """Stone game where you can only take adjacent piles"""
        # This changes the strategy significantly
        # For simplicity, use standard DP (actual problem would be different)
        return stone_game_dp_classic(piles)
    
    def stone_game_k_piles(piles, k):
        """Stone game where you can take up to k piles per turn"""
        n = len(piles)
        memo = {}
        
        def minimax(left, right, is_alice_turn):
            if left > right:
                return 0
            
            state = (left, right, is_alice_turn)
            if state in memo:
                return memo[state]
            
            if is_alice_turn:
                # Alice maximizes
                best = float('-inf')
                for take in range(1, min(k + 1, right - left + 2)):
                    # Take from left
                    if left + take - 1 <= right:
                        score = sum(piles[left:left + take])
                        best = max(best, score + minimax(left + take, right, False))
                    
                    # Take from right
                    if right - take + 1 >= left:
                        score = sum(piles[right - take + 1:right + 1])
                        best = max(best, score + minimax(left, right - take, False))
            else:
                # Bob minimizes (from Alice's perspective)
                best = float('inf')
                for take in range(1, min(k + 1, right - left + 2)):
                    # Take from left
                    if left + take - 1 <= right:
                        score = sum(piles[left:left + take])
                        best = min(best, minimax(left + take, right, True) - score)
                    
                    # Take from right
                    if right - take + 1 >= left:
                        score = sum(piles[right - take + 1:right + 1])
                        best = min(best, minimax(left, right - take, True) - score)
            
            memo[state] = best
            return best
        
        result = minimax(0, n - 1, True)
        return result > sum(piles) / 2
    
    # Test variants
    test_piles = [
        [5, 3, 8, 4],
        [3, 7, 2, 3],
        [1, 2, 3, 4, 5, 6],
        [10, 20, 30, 40]
    ]
    
    print("Stone Game Variants:")
    print("=" * 50)
    
    for piles in test_piles:
        print(f"\nPiles: {piles}")
        
        basic_result = stone_game_dp_classic(piles)
        print(f"Basic stone game: Alice wins: {basic_result}")
        
        # Multipliers variant
        multipliers = [1, 2, 1, 2][:len(piles)]
        mult_result = stone_game_with_multipliers(piles, multipliers)
        print(f"With multipliers {multipliers}: Alice wins: {mult_result}")
        
        # Three players (if applicable)
        if len(piles) >= 3:
            three_player_result = stone_game_three_players(piles)
            print(f"Three players: {three_player_result}")
        
        # K-piles variant
        if len(piles) <= 6:  # Only for small cases due to complexity
            k_result = stone_game_k_piles(piles, 2)
            print(f"Taking up to 2 piles: Alice wins: {k_result}")


# Test cases
def test_stone_game():
    """Test all implementations with various inputs"""
    test_cases = [
        ([5, 3, 8, 4], True),
        ([3, 7, 2, 3], True),
        ([1, 2], True),
        ([2, 1], True),
        ([1, 3, 5, 7], True),
        ([2, 4, 6, 8], True),
        ([100, 1, 1, 100], True),
        ([10, 20, 30, 40, 50, 60], True)
    ]
    
    print("Testing Stone Game Solutions:")
    print("=" * 70)
    
    for i, (piles, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"piles = {piles}")
        print(f"Expected: {expected}")
        
        mathematical = stone_game_mathematical_insight(piles)
        dp_classic = stone_game_dp_classic(piles)
        score_result, scores = stone_game_with_scores(piles)
        strategy_result, moves, final_scores = stone_game_with_strategy(piles)
        
        print(f"Mathematical:     {mathematical} {'✓' if mathematical == expected else '✗'}")
        print(f"DP Classic:       {dp_classic} {'✓' if dp_classic == expected else '✗'}")
        print(f"With Scores:      {score_result} {'✓' if score_result == expected else '✗'}")
        print(f"With Strategy:    {strategy_result} {'✓' if strategy_result == expected else '✗'}")
        
        if scores:
            alice_score, bob_score = scores
            print(f"Final Scores:     Alice {alice_score}, Bob {bob_score}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    stone_game_analysis([5, 3, 8, 4])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    stone_game_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MATHEMATICAL GUARANTEE: Alice always wins with even piles and odd total")
    print("2. POSITION STRATEGY: Alice can choose even or odd positions optimally")
    print("3. MINIMAX OPTIMIZATION: Each player minimizes opponent's advantage")
    print("4. PERFECT INFORMATION: Complete game state visibility for both players")
    print("5. ZERO-SUM NATURE: Total stones fixed, pure competition")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Game Theory: Two-player zero-sum game optimization")
    print("• Economic Competition: Resource allocation in competitive markets")
    print("• Strategic Planning: Optimal decision making with perfect information")
    print("• Algorithm Design: Minimax and game tree applications")
    print("• Competitive Analysis: Market share and competitive advantage")


if __name__ == "__main__":
    test_stone_game()


"""
STONE GAME - CLASSIC TWO-PLAYER ZERO-SUM GAME:
==============================================

This problem demonstrates a perfect example of Game Theory DP:
- Classic two-player zero-sum game with perfect information
- Mathematical insight reveals guaranteed optimal strategy
- Minimax optimization with interval dynamic programming
- Strategic position analysis and optimal play computation

KEY INSIGHTS:
============
1. **MATHEMATICAL GUARANTEE**: With even number of piles and odd total, Alice always wins
2. **POSITION STRATEGY**: Alice can choose to consistently take from even or odd positions
3. **MINIMAX OPTIMIZATION**: Each player plays to minimize opponent's maximum advantage
4. **PERFECT INFORMATION**: Both players have complete knowledge of all pile sizes
5. **ZERO-SUM NATURE**: Fixed total stones create pure competitive scenario

ALGORITHM APPROACHES:
====================

1. **Mathematical Insight**: O(1) time, O(1) space
   - Recognizes that Alice can always win under given constraints
   - Most elegant solution but requires deep game theory understanding

2. **Classic DP**: O(n²) time, O(n²) space
   - Standard minimax DP for educational and general cases
   - Works for any pile configuration

3. **Score Tracking**: O(n²) time, O(n²) space
   - Computes actual final scores for both players
   - Useful for understanding game dynamics

4. **Strategy Reconstruction**: O(n²) time, O(n²) space
   - Determines optimal move sequence
   - Essential for game AI implementation

MATHEMATICAL PROOF OF ALICE'S VICTORY:
=====================================
**Even Piles Property**: With even number of piles, Alice can control position parity

**Strategy Options**:
- **Even Strategy**: Alice always chooses to take from even-indexed positions
- **Odd Strategy**: Alice always chooses to take from odd-indexed positions

**Guarantee**: One of these strategies gives Alice more than half the total stones

**Proof Sketch**:
```
Total stones = odd number
Even positions sum + Odd positions sum = odd total
Therefore: Even sum ≠ Odd sum
Alice chooses the strategy corresponding to the larger sum
```

CORE MINIMAX DP ALGORITHM:
=========================
```python
def stoneGame(piles):
    n = len(piles)
    # dp[i][j] = max advantage (Alice - Bob) for piles[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Base case: single pile
    for i in range(n):
        dp[i][i] = piles[i]
    
    # Fill for increasing lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Take left: gain piles[i], opponent optimizes on rest
            take_left = piles[i] - dp[i + 1][j]
            
            # Take right: gain piles[j], opponent optimizes on rest  
            take_right = piles[j] - dp[i][j - 1]
            
            dp[i][j] = max(take_left, take_right)
    
    return dp[0][n - 1] > 0
```

POSITION ANALYSIS STRATEGY:
==========================
**Even/Odd Decomposition**: Split piles by position parity

**Strategic Insight**: 
```python
even_sum = sum(piles[i] for i in range(0, n, 2))  # positions 0,2,4,...
odd_sum = sum(piles[i] for i in range(1, n, 2))   # positions 1,3,5,...

# Alice can guarantee max(even_sum, odd_sum) stones
alice_guaranteed = max(even_sum, odd_sum)
bob_maximum = min(even_sum, odd_sum)

# Alice wins since total is odd: alice_guaranteed > bob_maximum
```

OPTIMAL PLAY DYNAMICS:
======================
**Turn Alternation**: Players alternate between maximizing and minimizing

**Recursive Structure**: 
- Alice's optimal choice depends on Bob's optimal response
- Bob's optimal choice depends on Alice's optimal response

**State Space**: All possible game states [i,j] representing remaining piles

GAME TREE ANALYSIS:
==================
**Branching Factor**: 2 (take left or right pile)
**Tree Depth**: n (number of piles)
**Total States**: O(n²) unique subproblems
**Memoization**: DP eliminates redundant subtree exploration

STRATEGIC IMPLICATIONS:
======================
**First Move Advantage**: Alice gets first choice, slight advantage
**Information Advantage**: Perfect information eliminates uncertainty
**Forced Optimality**: Both players must play optimally (no mistakes)
**Endgame**: Final moves become deterministic

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n²) for DP approaches
- n² possible intervals [i,j]
- Constant work per interval

**Space Complexity**: O(n²) for 2D DP, can be optimized to O(n)
- Only previous row needed for current computation

**Practical Performance**: Efficient for reasonable pile counts

VARIANTS AND EXTENSIONS:
=======================
**Different Constraints**: 
- Odd number of piles (removes Alice's guarantee)
- Even total stones (allows ties)
- Multiple players (complex game theory)

**Rule Modifications**:
- Take multiple piles per turn
- Different scoring mechanisms
- Time limits or other constraints

**Applications**:
- Resource allocation games
- Competitive bidding scenarios
- Market share competition
- Strategic decision making

RELATED PROBLEMS:
================
- **Predict the Winner**: General version without special constraints
- **Stone Game II**: Variable number of piles per turn
- **Stone Game III**: Different scoring mechanism
- **Nim Games**: Related combinatorial game theory

EDUCATIONAL VALUE:
=================
**Game Theory Concepts**:
- Zero-sum games
- Perfect information
- Minimax optimization
- Strategic dominance

**Algorithm Design**:
- Interval dynamic programming
- Recursive problem decomposition
- Optimal substructure
- State space analysis

This problem beautifully illustrates how mathematical insight
can sometimes provide more elegant solutions than algorithmic
approaches, while the DP solution demonstrates fundamental
game theory principles applicable to many competitive scenarios.
"""
