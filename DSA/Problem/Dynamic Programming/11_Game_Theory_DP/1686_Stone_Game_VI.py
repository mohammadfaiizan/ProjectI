"""
LeetCode 1686: Stone Game VI
Difficulty: Medium
Category: Game Theory DP - Dual Scoring Optimization

PROBLEM DESCRIPTION:
===================
Alice and Bob take turns playing a game, with Alice starting first.

There are n stones in a pile. On each player's turn, they can remove a stone from the pile and receive points based on the stone's value. Alice and Bob may value the same stone differently.

You are given two integer arrays of length n, aliceValues and bobValues. Each aliceValues[i] and bobValues[i] represents how much Alice and Bob value the ith stone, respectively.

The winner is the person with the most points after all the stones are chosen. If both players have the same number of points, the game results in a draw.

Assuming both players play optimally, return:
- 1 if Alice will win,
- -1 if Bob will win, or
- 0 if the game will result in a draw.

Example 1:
Input: aliceValues = [1,3], bobValues = [2,1]
Output: 1
Explanation:
If Alice takes stone 0, and Bob takes stone 1, then:
- Alice's score = 1
- Bob's score = 1
This is a draw.
If Alice takes stone 1, and Bob takes stone 0, then:
- Alice's score = 3
- Bob's score = 2
Alice wins.
Alice should take stone 1 first.

Example 2:
Input: aliceValues = [1,2], bobValues = [3,1]
Output: 0
Explanation:
If Alice takes stone 0, and Bob takes stone 1, then:
- Alice's score = 1
- Bob's score = 1
If Alice takes stone 1, and Bob takes stone 0, then:
- Alice's score = 2
- Bob's score = 3
Regardless of Alice's choice, Bob can make Alice never win.

Example 3:
Input: aliceValues = [2,4,3], bobValues = [1,6,7]
Output: -1

Constraints:
- n == aliceValues.length == bobValues.length
- 1 <= n <= 10^5
- 1 <= aliceValues[i], bobValues[i] <= 100
"""


def stone_game_vi_greedy(aliceValues, bobValues):
    """
    GREEDY APPROACH:
    ===============
    Sort stones by combined value (aliceValues[i] + bobValues[i]) in descending order.
    
    Time Complexity: O(n log n) - sorting
    Space Complexity: O(n) - for storing indices
    """
    n = len(aliceValues)
    
    # Create list of (combined_value, index) and sort by combined value descending
    stones = [(aliceValues[i] + bobValues[i], i) for i in range(n)]
    stones.sort(reverse=True)
    
    alice_score = 0
    bob_score = 0
    
    # Players alternate turns, Alice goes first
    for turn, (combined_value, idx) in enumerate(stones):
        if turn % 2 == 0:  # Alice's turn
            alice_score += aliceValues[idx]
        else:  # Bob's turn
            bob_score += bobValues[idx]
    
    if alice_score > bob_score:
        return 1
    elif bob_score > alice_score:
        return -1
    else:
        return 0


def stone_game_vi_with_analysis(aliceValues, bobValues):
    """
    STONE GAME VI WITH DETAILED ANALYSIS:
    ====================================
    Solve using greedy approach and provide comprehensive insights.
    
    Time Complexity: O(n log n) - sorting + analysis
    Space Complexity: O(n) - storing analysis data
    """
    n = len(aliceValues)
    
    analysis = {
        'alice_values': aliceValues[:],
        'bob_values': bobValues[:],
        'num_stones': n,
        'total_alice_value': sum(aliceValues),
        'total_bob_value': sum(bobValues),
        'stone_analysis': [],
        'optimal_order': [],
        'game_sequence': [],
        'strategy_insights': []
    }
    
    # Analyze each stone
    for i in range(n):
        stone_info = {
            'index': i,
            'alice_value': aliceValues[i],
            'bob_value': bobValues[i],
            'combined_value': aliceValues[i] + bobValues[i],
            'alice_advantage': aliceValues[i] - bobValues[i],
            'strategic_importance': aliceValues[i] + bobValues[i]  # How much total value is at stake
        }
        analysis['stone_analysis'].append(stone_info)
    
    # Sort by strategic importance (combined value)
    stones_by_importance = sorted(analysis['stone_analysis'], 
                                key=lambda x: x['combined_value'], reverse=True)
    
    analysis['optimal_order'] = [stone['index'] for stone in stones_by_importance]
    
    # Simulate optimal game
    alice_score = 0
    bob_score = 0
    
    for turn, stone in enumerate(stones_by_importance):
        move = {
            'turn': turn + 1,
            'player': 'Alice' if turn % 2 == 0 else 'Bob',
            'stone_index': stone['index'],
            'alice_value': stone['alice_value'],
            'bob_value': stone['bob_value'],
            'combined_value': stone['combined_value']
        }
        
        if turn % 2 == 0:  # Alice's turn
            alice_score += stone['alice_value']
            move['score_gained'] = stone['alice_value']
            move['denied_score'] = stone['bob_value']
        else:  # Bob's turn
            bob_score += stone['bob_value']
            move['score_gained'] = stone['bob_value']
            move['denied_score'] = stone['alice_value']
        
        move['alice_total'] = alice_score
        move['bob_total'] = bob_score
        analysis['game_sequence'].append(move)
    
    # Determine winner
    if alice_score > bob_score:
        winner = 1
        winner_name = "Alice"
    elif bob_score > alice_score:
        winner = -1
        winner_name = "Bob"
    else:
        winner = 0
        winner_name = "Draw"
    
    analysis['final_scores'] = {
        'alice': alice_score,
        'bob': bob_score,
        'difference': alice_score - bob_score,
        'winner': winner,
        'winner_name': winner_name
    }
    
    # Strategic insights
    analysis['strategy_insights'].append(f"Game result: {winner_name}")
    analysis['strategy_insights'].append(f"Final scores: Alice {alice_score}, Bob {bob_score}")
    analysis['strategy_insights'].append("Optimal strategy: Take stones with highest combined value first")
    
    # Analyze stone importance
    high_value_stones = [s for s in stones_by_importance if s['combined_value'] >= max(s['combined_value'] for s in stones_by_importance) * 0.8]
    analysis['strategy_insights'].append(f"High-priority stones: {len(high_value_stones)}/{n}")
    
    # Analyze advantages
    alice_favored = sum(1 for s in analysis['stone_analysis'] if s['alice_advantage'] > 0)
    bob_favored = sum(1 for s in analysis['stone_analysis'] if s['alice_advantage'] < 0)
    neutral = n - alice_favored - bob_favored
    
    analysis['strategy_insights'].append(f"Stone preferences: Alice favored {alice_favored}, Bob favored {bob_favored}, Neutral {neutral}")
    
    return winner, analysis


def stone_game_vi_analysis(aliceValues, bobValues):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze Stone Game VI with detailed strategic insights.
    """
    print(f"Stone Game VI Analysis:")
    print(f"Alice values: {aliceValues}")
    print(f"Bob values: {bobValues}")
    print(f"Number of stones: {len(aliceValues)}")
    
    # Basic greedy solution
    greedy_result = stone_game_vi_greedy(aliceValues, bobValues)
    result_text = ["Bob wins", "Draw", "Alice wins"][greedy_result + 1]
    print(f"Greedy approach result: {greedy_result} ({result_text})")
    
    # Detailed analysis
    detailed_result, analysis = stone_game_vi_with_analysis(aliceValues, bobValues)
    
    print(f"\nDetailed Game Analysis:")
    final = analysis['final_scores']
    print(f"Winner: {final['winner_name']}")
    print(f"Alice final score: {final['alice']}")
    print(f"Bob final score: {final['bob']}")
    print(f"Score difference: {final['difference']}")
    
    print(f"\nStone Analysis (by strategic importance):")
    for i, stone in enumerate(analysis['stone_analysis']):
        if stone['combined_value'] == max(s['combined_value'] for s in analysis['stone_analysis']):
            priority = "HIGH"
        elif stone['combined_value'] >= sum(s['combined_value'] for s in analysis['stone_analysis']) / len(analysis['stone_analysis']):
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        advantage = "Alice" if stone['alice_advantage'] > 0 else "Bob" if stone['alice_advantage'] < 0 else "Neutral"
        
        print(f"  Stone {stone['index']}: Alice={stone['alice_value']}, Bob={stone['bob_value']}, "
              f"Combined={stone['combined_value']}, Priority={priority}, Favors={advantage}")
    
    print(f"\nOptimal Game Sequence:")
    for move in analysis['game_sequence']:
        print(f"  Turn {move['turn']}: {move['player']} takes stone {move['stone_index']} "
              f"(gains {move['score_gained']}, denies {move['denied_score']}) "
              f"-> Alice: {move['alice_total']}, Bob: {move['bob_total']}")
    
    print(f"\nStrategy Insights:")
    for insight in analysis['strategy_insights']:
        print(f"  • {insight}")
    
    # Mathematical explanation
    print(f"\nMathematical Explanation:")
    print(f"• Key insight: Taking a stone gives you aliceValues[i] or bobValues[i]")
    print(f"• But it also DENIES opponent their value for that stone")
    print(f"• Net effect of taking stone i: your_value[i] + opponent_value[i]")
    print(f"• Therefore, prioritize stones with highest combined value")
    print(f"• This maximizes your gain + opponent's loss simultaneously")
    
    return detailed_result


def stone_game_vi_proof():
    """
    MATHEMATICAL PROOF OF GREEDY OPTIMALITY:
    =======================================
    Explain why the greedy approach is optimal.
    """
    print("Mathematical Proof of Greedy Optimality:")
    print("=" * 50)
    
    print("Theorem: Sorting by combined value (aliceValues[i] + bobValues[i]) gives optimal play.")
    print()
    
    print("Proof:")
    print("1. Consider any two stones i and j where combined_value[i] > combined_value[j]")
    print("2. Suppose Alice takes j first, then Bob takes i")
    print("   - Alice gets: aliceValues[j]")
    print("   - Bob gets: bobValues[i]")
    print("   - Net for Alice: aliceValues[j] - bobValues[i]")
    print()
    
    print("3. Alternative: Alice takes i first, then Bob takes j")
    print("   - Alice gets: aliceValues[i]")
    print("   - Bob gets: bobValues[j]")
    print("   - Net for Alice: aliceValues[i] - bobValues[j]")
    print()
    
    print("4. Difference between strategies:")
    print("   (aliceValues[i] - bobValues[j]) - (aliceValues[j] - bobValues[i])")
    print("   = aliceValues[i] + bobValues[i] - aliceValues[j] - bobValues[j]")
    print("   = combined_value[i] - combined_value[j]")
    print("   > 0 (since combined_value[i] > combined_value[j])")
    print()
    
    print("5. Therefore, Alice should always prioritize stones with higher combined value")
    print("6. Bob, playing optimally, will also prioritize remaining stones by combined value")
    print("7. This results in both players taking stones in order of decreasing combined value")
    print()
    
    print("Conclusion: The greedy algorithm is optimal! ✓")


def stone_game_vi_variants():
    """
    STONE GAME VI VARIANTS:
    ======================
    Different rule modifications and extensions.
    """
    
    def stone_game_weighted_values(aliceValues, bobValues, aliceWeight, bobWeight):
        """Variant where player values are weighted differently"""
        n = len(aliceValues)
        
        # Calculate effective combined values with weights
        stones = []
        for i in range(n):
            # Weighted combined value considers importance of denying opponent
            combined_value = aliceWeight * aliceValues[i] + bobWeight * bobValues[i]
            stones.append((combined_value, i))
        
        stones.sort(reverse=True)
        
        alice_score = 0
        bob_score = 0
        
        for turn, (_, idx) in enumerate(stones):
            if turn % 2 == 0:  # Alice's turn
                alice_score += aliceValues[idx]
            else:  # Bob's turn
                bob_score += bobValues[idx]
        
        if alice_score > bob_score:
            return 1
        elif bob_score > alice_score:
            return -1
        else:
            return 0
    
    def stone_game_three_players(aliceValues, bobValues, charlieValues):
        """Approximate three-player version"""
        n = len(aliceValues)
        
        # For three players, prioritize by total combined value
        stones = []
        for i in range(n):
            total_value = aliceValues[i] + bobValues[i] + charlieValues[i]
            stones.append((total_value, i))
        
        stones.sort(reverse=True)
        
        alice_score = 0
        bob_score = 0
        charlie_score = 0
        
        for turn, (_, idx) in enumerate(stones):
            if turn % 3 == 0:  # Alice's turn
                alice_score += aliceValues[idx]
            elif turn % 3 == 1:  # Bob's turn
                bob_score += bobValues[idx]
            else:  # Charlie's turn
                charlie_score += charlieValues[idx]
        
        scores = [alice_score, bob_score, charlie_score]
        max_score = max(scores)
        winners = [i for i, score in enumerate(scores) if score == max_score]
        
        if len(winners) == 1:
            return f"Player {winners[0]} wins with {max_score}"
        else:
            return f"Draw between players {winners} with {max_score}"
    
    def stone_game_with_costs(aliceValues, bobValues, costs):
        """Variant where taking stones has costs"""
        n = len(aliceValues)
        
        # Adjust values by subtracting costs
        adj_alice = [aliceValues[i] - costs[i] for i in range(n)]
        adj_bob = [bobValues[i] - costs[i] for i in range(n)]
        
        # Use combined value of adjusted values
        stones = []
        for i in range(n):
            if adj_alice[i] + adj_bob[i] > 0:  # Only consider profitable stones
                combined_value = adj_alice[i] + adj_bob[i]
                stones.append((combined_value, i))
        
        stones.sort(reverse=True)
        
        alice_score = 0
        bob_score = 0
        
        for turn, (_, idx) in enumerate(stones):
            if turn % 2 == 0:  # Alice's turn
                alice_score += adj_alice[idx]
            else:  # Bob's turn
                bob_score += adj_bob[idx]
        
        if alice_score > bob_score:
            return 1
        elif bob_score > alice_score:
            return -1
        else:
            return 0
    
    def stone_game_with_limited_picks(aliceValues, bobValues, max_picks):
        """Variant where each player can only pick limited number of stones"""
        n = len(aliceValues)
        
        # Sort by combined value
        stones = [(aliceValues[i] + bobValues[i], i) for i in range(n)]
        stones.sort(reverse=True)
        
        alice_score = 0
        bob_score = 0
        alice_picks = 0
        bob_picks = 0
        
        for turn, (_, idx) in enumerate(stones):
            if turn % 2 == 0:  # Alice's turn
                if alice_picks < max_picks:
                    alice_score += aliceValues[idx]
                    alice_picks += 1
            else:  # Bob's turn
                if bob_picks < max_picks:
                    bob_score += bobValues[idx]
                    bob_picks += 1
            
            # Stop if both players reached their limit
            if alice_picks >= max_picks and bob_picks >= max_picks:
                break
        
        if alice_score > bob_score:
            return 1
        elif bob_score > alice_score:
            return -1
        else:
            return 0
    
    # Test variants
    test_cases = [
        ([1, 3], [2, 1]),
        ([1, 2], [3, 1]),
        ([2, 4, 3], [1, 6, 7]),
        ([1, 3, 7], [4, 2, 1])
    ]
    
    print("Stone Game VI Variants:")
    print("=" * 50)
    
    for aliceValues, bobValues in test_cases:
        print(f"\nAlice values: {aliceValues}")
        print(f"Bob values: {bobValues}")
        
        basic_result = stone_game_vi_greedy(aliceValues, bobValues)
        result_names = ["Bob wins", "Draw", "Alice wins"]
        print(f"Basic Stone Game VI: {result_names[basic_result + 1]}")
        
        # Weighted variant
        weighted_result = stone_game_weighted_values(aliceValues, bobValues, 1.5, 1.0)
        print(f"Weighted (Alice 1.5x): {result_names[weighted_result + 1]}")
        
        # Three players variant
        if len(aliceValues) >= 3:
            charlieValues = [2] * len(aliceValues)  # Charlie values all stones equally
            three_player_result = stone_game_three_players(aliceValues, bobValues, charlieValues)
            print(f"Three players: {three_player_result}")
        
        # With costs variant
        costs = [1] * len(aliceValues)  # Uniform cost
        cost_result = stone_game_with_costs(aliceValues, bobValues, costs)
        print(f"With costs: {result_names[cost_result + 1]}")
        
        # Limited picks variant
        max_picks = len(aliceValues) // 2 + 1
        limited_result = stone_game_with_limited_picks(aliceValues, bobValues, max_picks)
        print(f"Limited to {max_picks} picks: {result_names[limited_result + 1]}")


# Test cases
def test_stone_game_vi():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 3], [2, 1], 1),
        ([1, 2], [3, 1], 0),
        ([2, 4, 3], [1, 6, 7], -1),
        ([1], [2], -1),
        ([2], [1], 1),
        ([1, 3, 7], [4, 2, 1], 1)
    ]
    
    print("Testing Stone Game VI Solutions:")
    print("=" * 70)
    
    for i, (aliceValues, bobValues, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"aliceValues = {aliceValues}")
        print(f"bobValues = {bobValues}")
        print(f"Expected: {expected}")
        
        greedy_result = stone_game_vi_greedy(aliceValues, bobValues)
        detailed_result, _ = stone_game_vi_with_analysis(aliceValues, bobValues)
        
        print(f"Greedy:           {greedy_result:>2} {'✓' if greedy_result == expected else '✗'}")
        print(f"With Analysis:    {detailed_result:>2} {'✓' if detailed_result == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    stone_game_vi_analysis([2, 4, 3], [1, 6, 7])
    
    # Mathematical proof
    print(f"\n" + "=" * 70)
    stone_game_vi_proof()
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    stone_game_vi_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. COMBINED VALUE STRATEGY: Prioritize stones with highest aliceValues[i] + bobValues[i]")
    print("2. DUAL OPTIMIZATION: Taking a stone gains your value AND denies opponent's value")
    print("3. GREEDY OPTIMALITY: Simple greedy approach is mathematically optimal")
    print("4. ZERO-SUM NATURE: Focus on net advantage rather than absolute scores")
    print("5. STRATEGIC DENIAL: Sometimes taking a stone to deny opponent is optimal")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Resource Competition: Optimal strategy in competitive resource allocation")
    print("• Market Economics: Bidding strategies when values differ between parties")
    print("• Game Theory: Two-player zero-sum games with different utility functions")
    print("• Auction Theory: Optimal bidding when opponents have different valuations")
    print("• Strategic Planning: Competitive decision making with denial considerations")


if __name__ == "__main__":
    test_stone_game_vi()


"""
STONE GAME VI - DUAL SCORING OPTIMIZATION WITH GREEDY STRATEGY:
===============================================================

This problem demonstrates elegant dual scoring optimization:
- Players have different valuations for the same resources
- Optimal strategy considers both gaining your value AND denying opponent's value
- Surprisingly, a simple greedy approach is mathematically optimal
- Showcases how game theory can have elegant closed-form solutions

KEY INSIGHTS:
============
1. **DUAL SCORING**: Each stone has different values for Alice and Bob
2. **COMBINED VALUE STRATEGY**: Optimal to prioritize stones by aliceValues[i] + bobValues[i]
3. **DENIAL CONSIDERATION**: Taking a stone denies opponent their value for it
4. **GREEDY OPTIMALITY**: Simple greedy algorithm is mathematically proven optimal
5. **ZERO-SUM ESSENCE**: Net effect focuses on relative advantage maximization

ALGORITHM APPROACHES:
====================

1. **Greedy Strategy**: O(n log n) time, O(n) space
   - Sort stones by combined value in descending order
   - Players alternate taking stones in this order
   - Mathematically proven optimal

2. **Analysis Version**: O(n log n) time, O(n) space
   - Include detailed strategic analysis and game reconstruction
   - Provides insights into why greedy approach works

CORE GREEDY ALGORITHM:
=====================
```python
def stoneGameVI(aliceValues, bobValues):
    n = len(aliceValues)
    
    # Sort stones by combined value (descending)
    stones = sorted(range(n), 
                   key=lambda i: aliceValues[i] + bobValues[i], 
                   reverse=True)
    
    alice_score = 0
    bob_score = 0
    
    # Players alternate, Alice goes first
    for turn, stone_idx in enumerate(stones):
        if turn % 2 == 0:  # Alice's turn
            alice_score += aliceValues[stone_idx]
        else:  # Bob's turn
            bob_score += bobValues[stone_idx]
    
    if alice_score > bob_score:
        return 1    # Alice wins
    elif bob_score > alice_score:
        return -1   # Bob wins
    else:
        return 0    # Draw
```

MATHEMATICAL PROOF OF OPTIMALITY:
=================================
**Theorem**: Greedy strategy (sorting by combined value) is optimal.

**Proof**: Consider any two stones i and j where combined_value[i] > combined_value[j].

**Case Analysis**:
- If Alice takes j then Bob takes i: Alice nets `aliceValues[j] - bobValues[i]`
- If Alice takes i then Bob takes j: Alice nets `aliceValues[i] - bobValues[j]`

**Difference**: 
```
(aliceValues[i] - bobValues[j]) - (aliceValues[j] - bobValues[i])
= aliceValues[i] + bobValues[i] - aliceValues[j] - bobValues[j]
= combined_value[i] - combined_value[j] > 0
```

Therefore, Alice should always prefer stones with higher combined value.

STRATEGIC INTUITION:
===================
**Dual Effect**: Taking stone i gives you:
1. **Direct Gain**: `your_value[i]` points
2. **Denial Effect**: Opponent loses potential `opponent_value[i]` points

**Net Advantage**: `your_value[i] + opponent_value[i]` = total strategic value

**Optimal Strategy**: Maximize net advantage by taking stones with highest combined value

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n log n) - dominated by sorting
**Space Complexity**: O(n) - for storing stone indices
**Practical Performance**: Very efficient, much better than exponential game tree approaches

DUAL SCORING DYNAMICS:
======================
**Value Asymmetry**: Same resource valued differently by each player
**Strategic Priorities**: 
- High alice_value + Low bob_value: Alice strongly favors
- Low alice_value + High bob_value: Alice takes to deny Bob
- High alice_value + High bob_value: Critical stones both want

**Optimal Play Pattern**: Both players follow same sorting order

GAME THEORY CONNECTIONS:
=======================
**Zero-Sum Nature**: Total value fixed, pure competition
**Nash Equilibrium**: Greedy strategy is mutual best response
**Dominant Strategy**: Combined value sorting dominates all other strategies

APPLICATIONS:
============
- **Auction Theory**: Bidding when participants have different valuations
- **Resource Competition**: Competitive allocation with varying utilities
- **Market Economics**: Strategic resource acquisition
- **Competitive Programming**: Elegant solution to complex-seeming problem
- **Game Design**: Balanced competitive mechanics

RELATED PROBLEMS:
================
- **Job Scheduling**: Maximize profit while denying competitors
- **Resource Allocation**: Competitive distribution scenarios
- **Auction Design**: Optimal bidding strategies
- **Market Analysis**: Competitive advantage calculation

VARIANTS:
========
- **Weighted Values**: Different importance weights for denial vs gain
- **Multi-Player**: Extension to more than two players
- **Limited Picks**: Constraints on number of stones each player can take
- **Costs**: Taking stones has associated costs

EDGE CASES:
==========
- **Single Stone**: Trivial comparison of values
- **Equal Combined Values**: Tie-breaking may affect outcome
- **Dominated Strategies**: When one player's values are always higher
- **Zero Values**: Handling stones with no value to some players

OPTIMIZATION TECHNIQUES:
=======================
**Sorting Optimization**: Use stable sort for consistent tie-breaking
**Early Termination**: In some variants, can determine winner early
**Memory Efficiency**: In-place operations where possible
**Preprocessing**: Calculate combined values once

This problem beautifully demonstrates how sophisticated game
theory problems can sometimes have surprisingly simple optimal
solutions, showcasing the power of mathematical analysis in
competitive algorithm design and strategic decision making.
"""
