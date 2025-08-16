"""
LeetCode 837: New 21 Game
Difficulty: Medium
Category: Probability DP - Conditional Probability

PROBLEM DESCRIPTION:
===================
Alice plays the following game, loosely based on the card game "21".

Alice starts with 0 points and draws numbers while she has less than k points. During each draw, she gains an integer number of points in the range [1, maxPts] where each integer is equally likely.

Once a player reaches k or more points, they can no longer draw numbers.

Return the probability that Alice has n or fewer points.

Example 1:
Input: n = 10, k = 1, maxPts = 10
Output: 1.0
Explanation: Alice gets a single card, then stops.

Example 2:
Input: n = 6, k = 1, maxPts = 10
Output: 0.6
Explanation: Alice gets a single card, then stops.
Since each card has equal probability, she has a 6/10 = 0.6 probability to get a card with 1-6 points.

Example 3:
Input: n = 21, k = 17, maxPts = 10
Output: 0.73278

Constraints:
- 0 <= k <= n <= 10^4
- 1 <= maxPts <= 10^4
"""


def new_21_game_recursive(n, k, maxPts):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to calculate probability directly.
    
    Time Complexity: O(maxPts^k) - exponential without memoization
    Space Complexity: O(k) - recursion stack
    """
    def probability(points):
        # Base cases
        if points >= k:
            # Can't draw anymore, check if points <= n
            return 1.0 if points <= n else 0.0
        
        # If already over n, impossible to win
        if points > n:
            return 0.0
        
        # Calculate probability by trying all possible draws
        total_prob = 0.0
        for draw in range(1, maxPts + 1):
            total_prob += probability(points + draw)
        
        return total_prob / maxPts
    
    return probability(0)


def new_21_game_memoization(n, k, maxPts):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed states.
    
    Time Complexity: O(n * maxPts) - each state computed once
    Space Complexity: O(n) - memoization table
    """
    memo = {}
    
    def probability(points):
        if points >= k:
            return 1.0 if points <= n else 0.0
        
        if points > n:
            return 0.0
        
        if points in memo:
            return memo[points]
        
        total_prob = 0.0
        for draw in range(1, maxPts + 1):
            total_prob += probability(points + draw)
        
        result = total_prob / maxPts
        memo[points] = result
        return result
    
    return probability(0)


def new_21_game_dp_bottom_up(n, k, maxPts):
    """
    BOTTOM-UP DP APPROACH:
    =====================
    Build solution from larger points to smaller points.
    
    Time Complexity: O(n * maxPts) - nested loops
    Space Complexity: O(n) - DP array
    """
    if k == 0:
        return 1.0 if n >= 0 else 0.0
    
    # dp[i] = probability of winning starting with i points
    dp = [0.0] * (n + maxPts + 1)
    
    # Base cases: points >= k
    for i in range(k, min(n + 1, n + maxPts + 1)):
        dp[i] = 1.0  # Win if points in [k, n]
    
    # Fill DP table from k-1 down to 0
    for points in range(k - 1, -1, -1):
        if points > n:
            dp[points] = 0.0
        else:
            total_prob = 0.0
            for draw in range(1, maxPts + 1):
                if points + draw < len(dp):
                    total_prob += dp[points + draw]
            dp[points] = total_prob / maxPts
    
    return dp[0]


def new_21_game_optimized(n, k, maxPts):
    """
    OPTIMIZED DP WITH SLIDING WINDOW:
    ================================
    Use sliding window to optimize the sum calculation.
    
    Time Complexity: O(n + k) - linear in most cases
    Space Complexity: O(n + k) - DP array
    """
    if k == 0:
        return 1.0 if n >= 0 else 0.0
    
    # dp[i] = probability of getting <= n points starting with i points
    dp = [0.0] * (n + maxPts + 1)
    
    # Base cases: if points >= k, can't draw anymore
    for i in range(k, min(n + 1, len(dp))):
        dp[i] = 1.0
    
    # Calculate sum for sliding window
    window_sum = sum(dp[k:k + maxPts]) if k + maxPts <= len(dp) else sum(dp[k:])
    
    # Fill DP table from k-1 down to 0
    for points in range(k - 1, -1, -1):
        dp[points] = window_sum / maxPts
        
        # Update sliding window for next iteration
        if points + maxPts < len(dp):
            window_sum += dp[points + maxPts]
        if points + maxPts < len(dp):
            window_sum -= dp[points + maxPts + 1] if points + maxPts + 1 < len(dp) else 0
    
    return dp[0]


def new_21_game_with_analysis(n, k, maxPts):
    """
    NEW 21 GAME WITH DETAILED ANALYSIS:
    ===================================
    Solve the game and provide comprehensive insights.
    
    Time Complexity: O(n * maxPts) - DP computation + analysis
    Space Complexity: O(n) - DP array + analysis data
    """
    analysis = {
        'n': n,
        'k': k,
        'maxPts': maxPts,
        'game_phases': {},
        'probability_distribution': {},
        'expected_values': {},
        'insights': []
    }
    
    if k == 0:
        analysis['insights'].append("k=0: Alice never draws, wins if n>=0")
        return (1.0 if n >= 0 else 0.0), analysis
    
    # DP calculation with detailed tracking
    dp = [0.0] * (n + maxPts + 1)
    
    # Analyze different game phases
    analysis['game_phases'] = {
        'drawing_phase': list(range(0, k)),
        'terminal_winning': list(range(k, min(n + 1, len(dp)))),
        'terminal_losing': list(range(n + 1, len(dp)))
    }
    
    # Base cases
    for i in range(k, min(n + 1, len(dp))):
        dp[i] = 1.0
    
    # Track probability evolution
    prob_evolution = []
    
    for points in range(k - 1, -1, -1):
        if points > n:
            dp[points] = 0.0
        else:
            total_prob = 0.0
            draws_analysis = {}
            
            for draw in range(1, maxPts + 1):
                if points + draw < len(dp):
                    draws_analysis[draw] = {
                        'resulting_points': points + draw,
                        'probability': dp[points + draw]
                    }
                    total_prob += dp[points + draw]
            
            dp[points] = total_prob / maxPts
            
            prob_evolution.append({
                'points': points,
                'probability': dp[points],
                'draws_analysis': draws_analysis
            })
    
    analysis['probability_distribution'] = {i: dp[i] for i in range(min(len(dp), n + 1))}
    
    # Calculate expected final points
    expected_final_points = 0.0
    point_distribution = {}
    
    # This is complex to calculate exactly, so we'll provide approximations
    analysis['expected_values'] = {
        'win_probability': dp[0],
        'lose_probability': 1 - dp[0]
    }
    
    # Generate insights
    win_prob = dp[0]
    analysis['insights'].append(f"Probability of winning (≤{n} points): {win_prob:.6f}")
    analysis['insights'].append(f"Probability of losing (>{n} points): {1-win_prob:.6f}")
    
    # Analyze game parameters
    min_final_points = k
    max_final_points = k - 1 + maxPts
    analysis['insights'].append(f"Final points range: [{min_final_points}, {max_final_points}]")
    
    if n < k:
        analysis['insights'].append("Impossible: n < k means Alice must lose")
    elif n >= max_final_points:
        analysis['insights'].append("Guaranteed win: n ≥ maximum possible final points")
    
    # Risk analysis
    danger_zone = max_final_points - n
    if danger_zone > 0:
        analysis['insights'].append(f"Danger zone: {danger_zone} points above target")
    
    # Expected number of draws
    expected_draws = (k - 1 + maxPts) / 2  # Rough approximation
    analysis['insights'].append(f"Expected draws: ~{expected_draws:.1f}")
    
    return dp[0], analysis


def new_21_game_analysis(n, k, maxPts):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the New 21 Game with detailed insights.
    """
    print(f"New 21 Game Analysis:")
    print(f"Target (n): {n}")
    print(f"Stop threshold (k): {k}")
    print(f"Max points per draw (maxPts): {maxPts}")
    
    # Basic game analysis
    if k == 0:
        print("Special case: k=0, Alice never draws")
        return 1.0 if n >= 0 else 0.0
    
    min_final = k
    max_final = k - 1 + maxPts
    print(f"Possible final points: [{min_final}, {max_final}]")
    print(f"Winning range: [{k}, {n}]")
    print(f"Losing range: [{n+1}, {max_final}]")
    
    # Different approaches
    if k <= 10 and maxPts <= 10:
        try:
            recursive = new_21_game_recursive(n, k, maxPts)
            print(f"Recursive result: {recursive:.6f}")
        except:
            print("Recursive: Too slow")
    
    memoization = new_21_game_memoization(n, k, maxPts)
    bottom_up = new_21_game_dp_bottom_up(n, k, maxPts)
    optimized = new_21_game_optimized(n, k, maxPts)
    
    print(f"Memoization result: {memoization:.6f}")
    print(f"Bottom-up DP result: {bottom_up:.6f}")
    print(f"Optimized result: {optimized:.6f}")
    
    # Detailed analysis
    detailed_result, analysis = new_21_game_with_analysis(n, k, maxPts)
    
    print(f"\nDetailed Analysis:")
    print(f"Win probability: {detailed_result:.6f}")
    
    print(f"\nGame Phases:")
    phases = analysis['game_phases']
    print(f"Drawing phase (can draw): {phases['drawing_phase']}")
    print(f"Terminal winning states: {phases['terminal_winning']}")
    print(f"Terminal losing states: {phases['terminal_losing']}")
    
    print(f"\nInsights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    # Show probability distribution for small cases
    if len(analysis['probability_distribution']) <= 20:
        print(f"\nProbability Distribution:")
        for points, prob in analysis['probability_distribution'].items():
            if prob > 0.001:  # Only show significant probabilities
                print(f"  P(win from {points} points) = {prob:.4f}")
    
    return detailed_result


def new_21_game_variants():
    """
    NEW 21 GAME VARIANTS:
    ====================
    Different game rule modifications.
    """
    
    def new_21_game_non_uniform(n, k, draw_probs):
        """Game with non-uniform draw probabilities"""
        memo = {}
        
        def probability(points):
            if points >= k:
                return 1.0 if points <= n else 0.0
            
            if points > n:
                return 0.0
            
            if points in memo:
                return memo[points]
            
            total_prob = 0.0
            for draw, prob in enumerate(draw_probs, 1):
                if prob > 0:
                    total_prob += prob * probability(points + draw)
            
            memo[points] = total_prob
            return total_prob
        
        return probability(0)
    
    def new_21_game_with_penalty(n, k, maxPts, penalty_threshold):
        """Game where going over penalty_threshold has additional cost"""
        memo = {}
        
        def probability(points):
            if points >= k:
                if points <= n:
                    return 1.0 if points <= penalty_threshold else 0.5  # Penalty
                else:
                    return 0.0
            
            if points > n:
                return 0.0
            
            if points in memo:
                return memo[points]
            
            total_prob = 0.0
            for draw in range(1, maxPts + 1):
                total_prob += probability(points + draw)
            
            result = total_prob / maxPts
            memo[points] = result
            return result
        
        return probability(0)
    
    def new_21_game_multi_round(n, k, maxPts, rounds):
        """Game played over multiple rounds"""
        # Simplified: probability of winning at least one round
        single_round_prob = new_21_game_optimized(n, k, maxPts)
        lose_all_rounds = (1 - single_round_prob) ** rounds
        return 1 - lose_all_rounds
    
    def new_21_game_expected_final_points(n, k, maxPts):
        """Calculate expected final points"""
        memo = {}
        
        def expected_points(points):
            if points >= k:
                return points  # Game ends, return current points
            
            if points in memo:
                return memo[points]
            
            total_expected = 0.0
            for draw in range(1, maxPts + 1):
                total_expected += expected_points(points + draw)
            
            result = total_expected / maxPts
            memo[points] = result
            return result
        
        return expected_points(0)
    
    # Test variants
    test_cases = [
        (10, 1, 10),
        (6, 1, 10),
        (21, 17, 10),
        (15, 10, 5)
    ]
    
    print("New 21 Game Variants:")
    print("=" * 50)
    
    for n, k, maxPts in test_cases:
        print(f"\nn={n}, k={k}, maxPts={maxPts}")
        
        basic_prob = new_21_game_optimized(n, k, maxPts)
        print(f"Basic game: {basic_prob:.6f}")
        
        # Non-uniform draws (favor smaller numbers)
        if maxPts <= 6:
            draw_probs = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03][:maxPts]
            draw_probs = [p / sum(draw_probs) for p in draw_probs]  # Normalize
            non_uniform_prob = new_21_game_non_uniform(n, k, draw_probs)
            print(f"Non-uniform draws: {non_uniform_prob:.6f}")
        
        # With penalty
        penalty_threshold = n - (n - k) // 2
        penalty_prob = new_21_game_with_penalty(n, k, maxPts, penalty_threshold)
        print(f"With penalty: {penalty_prob:.6f}")
        
        # Multi-round
        multi_round_prob = new_21_game_multi_round(n, k, maxPts, 3)
        print(f"3 rounds (win ≥1): {multi_round_prob:.6f}")
        
        # Expected final points
        if k <= 20 and maxPts <= 10:
            expected_final = new_21_game_expected_final_points(n, k, maxPts)
            print(f"Expected final points: {expected_final:.2f}")


# Test cases
def test_new_21_game():
    """Test all implementations with various inputs"""
    test_cases = [
        (10, 1, 10, 1.0),
        (6, 1, 10, 0.6),
        (21, 17, 10, 0.73278),
        (0, 0, 1, 1.0),
        (5, 3, 2, 1.0)
    ]
    
    print("Testing New 21 Game Solutions:")
    print("=" * 70)
    
    for i, (n, k, maxPts, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}, k = {k}, maxPts = {maxPts}")
        print(f"Expected: {expected}")
        
        # Skip recursive for large cases
        if k <= 8 and maxPts <= 8:
            try:
                recursive = new_21_game_recursive(n, k, maxPts)
                diff = abs(recursive - expected)
                print(f"Recursive:        {recursive:.6f} {'✓' if diff < 0.001 else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = new_21_game_memoization(n, k, maxPts)
        bottom_up = new_21_game_dp_bottom_up(n, k, maxPts)
        optimized = new_21_game_optimized(n, k, maxPts)
        
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
    new_21_game_analysis(21, 17, 10)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    new_21_game_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. CONDITIONAL PROBABILITY: Probability depends on current state and future draws")
    print("2. STOPPING CONDITION: Game structure with mandatory stopping at threshold k")
    print("3. UNIFORM DRAWS: Each draw value equally likely in basic version")
    print("4. OPTIMAL SUBSTRUCTURE: Probability from state depends on optimal from next states")
    print("5. BOUNDARY CONDITIONS: Win/lose determined by final point total")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Card Game Analysis: Probability calculations for card-based games")
    print("• Risk Assessment: Sequential decision making with stopping conditions")
    print("• Gambling Theory: Expected value analysis in betting scenarios")
    print("• Stochastic Processes: Discrete-time Markov chains with absorption")
    print("• Game Development: Balancing game mechanics and difficulty")


if __name__ == "__main__":
    test_new_21_game()


"""
NEW 21 GAME - CONDITIONAL PROBABILITY WITH STOPPING CONDITIONS:
===============================================================

This problem demonstrates probability DP with stopping conditions:
- Sequential random draws with threshold-based termination
- Conditional probability calculation with absorbing states
- Expected value analysis under constraint satisfaction
- Game theory elements with risk-reward trade-offs

KEY INSIGHTS:
============
1. **CONDITIONAL PROBABILITY**: Win probability depends on current points and draw distribution
2. **STOPPING CONDITIONS**: Mandatory termination at threshold k creates absorbing states
3. **UNIFORM DISTRIBUTION**: Each draw value [1, maxPts] equally likely
4. **BOUNDARY ANALYSIS**: Win/lose regions determined by final total vs target n
5. **RISK-REWARD DYNAMICS**: Balance between reaching k and not exceeding n

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(maxPts^k) time, O(k) space
   - Direct probability tree exploration
   - Exponential complexity without memoization

2. **Memoization**: O(n × maxPts) time, O(n) space
   - Top-down DP with state caching
   - Each point total computed once

3. **Bottom-up DP**: O(n × maxPts) time, O(n) space
   - Iterative DP from terminal states backward
   - Most straightforward implementation

4. **Sliding Window**: O(n + k) time, O(n) space
   - Optimized sum calculation using sliding window
   - Optimal complexity for this specific problem

CORE PROBABILITY DP ALGORITHM:
=============================
```python
def new21Game(n, k, maxPts):
    if k == 0:
        return 1.0 if n >= 0 else 0.0
    
    # dp[i] = probability of winning starting with i points
    dp = [0.0] * (n + maxPts + 1)
    
    # Base cases: points >= k (can't draw anymore)
    for i in range(k, min(n + 1, len(dp))):
        dp[i] = 1.0  # Win if final points in [k, n]
    
    # Fill DP table from k-1 down to 0
    for points in range(k - 1, -1, -1):
        total_prob = 0.0
        for draw in range(1, maxPts + 1):
            if points + draw < len(dp):
                total_prob += dp[points + draw]
        dp[points] = total_prob / maxPts
    
    return dp[0]
```

GAME STRUCTURE ANALYSIS:
=======================
**Phases**:
1. **Drawing Phase** (0 ≤ points < k): Alice must draw
2. **Terminal Phase** (points ≥ k): Game ends, win/lose determined

**State Transitions**: From points p < k, transition to p+i with probability 1/maxPts for i ∈ [1, maxPts]

**Absorbing States**: All points ≥ k are absorbing with deterministic outcomes

SLIDING WINDOW OPTIMIZATION:
============================
**Key Insight**: dp[points] = (sum of dp[points+1] to dp[points+maxPts]) / maxPts

**Sliding Window**: Maintain sum of relevant future states
```python
window_sum = sum(dp[k:k + maxPts])
for points in range(k-1, -1, -1):
    dp[points] = window_sum / maxPts
    # Update window: add dp[points+maxPts], remove dp[points+maxPts+1]
```

**Complexity Reduction**: From O(n × maxPts) to O(n + k)

PROBABILITY BOUNDARY ANALYSIS:
==============================
**Winning Region**: [k, n] - reach k or more without exceeding n
**Losing Region**: [n+1, k+maxPts-1] - exceed n before next draw
**Impossible Region**: Points ≥ k+maxPts cannot be reached

**Critical Cases**:
- n < k: Impossible to win (must reach k but can't exceed n)
- n ≥ k+maxPts-1: Guaranteed win (can't exceed n from any drawing state)

EXPECTED VALUE CALCULATIONS:
===========================
**Win Probability**: P(final points ≤ n)
**Expected Final Points**: E[points | game ends]
**Expected Number of Draws**: E[draws until points ≥ k]

**Risk Metrics**: Probability of exceeding target by various amounts

APPLICATIONS:
============
- **Card Game Design**: Balancing risk-reward in drawing games
- **Risk Assessment**: Sequential decisions with stopping thresholds
- **Gambling Analysis**: Expected value in threshold-based betting
- **Quality Control**: Acceptance sampling with accumulating evidence
- **Resource Management**: Sequential resource allocation with limits

RELATED PROBLEMS:
================
- **Random Walk with Barriers**: Similar stopping condition problems
- **Secretary Problem**: Optimal stopping theory
- **Gambler's Ruin**: Classic probability with absorbing boundaries
- **Optimal Stopping**: General class of threshold-based decisions

VARIANTS:
========
- **Non-uniform Draws**: Weighted probability distributions
- **Multiple Rounds**: Best-of-n game scenarios
- **Penalty Zones**: Additional costs for certain final totals
- **Variable Thresholds**: Dynamic k based on current points

EDGE CASES:
==========
- **k = 0**: Alice never draws, trivial win condition
- **k = n**: Must hit exactly k to win
- **maxPts = 1**: Deterministic game progression
- **Large n**: Win probability approaches certainty

OPTIMIZATION TECHNIQUES:
=======================
**Sliding Window**: Linear time complexity optimization
**Early Termination**: Stop when probability becomes negligible
**Symmetry Exploitation**: Use game symmetries to reduce computation
**Approximation Methods**: Monte Carlo simulation for validation

This problem beautifully demonstrates how stopping conditions
create absorbing Markov chains, requiring careful analysis
of boundary conditions and probability flow to compute
exact win probabilities in constrained stochastic games.
"""
