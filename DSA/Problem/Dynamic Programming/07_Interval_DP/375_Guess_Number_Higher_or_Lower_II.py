"""
LeetCode 375: Guess Number Higher or Lower II
Difficulty: Medium
Category: Interval DP - Game Theory

PROBLEM DESCRIPTION:
===================
We are playing the Guessing Game. The game will work as follows:
1. I pick a number between 1 and n.
2. You guess a number.
3. If you guess the wrong number, I will tell you whether the number I picked is higher or lower than your guess.
4. You will pay the amount equal to your guess.
5. The game continues until you guess the correct number.

Given a particular n, return the minimum amount of money you need to guarantee a win regardless of what number I pick.

Example 1:
Input: n = 10
Output: 16
Explanation: The winning strategy is:
- 1st round: Guess 7, your loss is 7.
- If the number is higher, guess 9, your loss is 9.
- If the number is lower, guess 3, your loss is 3.
- If the number is lower, guess 1, your loss is 1.
- If the number is higher, guess 2, your loss is 2.
The worst case in all these scenarios is that you pay 16.

Example 2:
Input: n = 1
Output: 0
Explanation: There is only one possible number, so you can guess it right away.

Example 3:
Input: n = 2
Output: 1
Explanation: There are two possible numbers, 1 and 2.
- Guess 1, your loss is 1.
- If the number is higher, guess 2, your loss is 2.
The worst case is that you pay 1.

Constraints:
- 1 <= n <= 200
"""

def get_money_amount_recursive(n):
    """
    RECURSIVE APPROACH:
    ==================
    Try all possible guesses and take the best worst-case scenario.
    
    Time Complexity: O(n!) - exponential branching
    Space Complexity: O(n) - recursion depth
    """
    def min_cost(start, end):
        if start >= end:
            return 0
        
        min_worst_case = float('inf')
        
        # Try each possible guess
        for guess in range(start, end + 1):
            # Cost of this guess
            cost = guess
            
            # Worst case: take maximum of both branches
            left_cost = min_cost(start, guess - 1)  # Number is lower
            right_cost = min_cost(guess + 1, end)   # Number is higher
            worst_case = cost + max(left_cost, right_cost)
            
            min_worst_case = min(min_worst_case, worst_case)
        
        return min_worst_case
    
    return min_cost(1, n)


def get_money_amount_memoization(n):
    """
    MEMOIZATION APPROACH:
    ====================
    Cache results for different ranges.
    
    Time Complexity: O(n^3) - n^2 states, O(n) transitions each
    Space Complexity: O(n^2) - memo table
    """
    memo = {}
    
    def min_cost(start, end):
        if start >= end:
            return 0
        
        if (start, end) in memo:
            return memo[(start, end)]
        
        min_worst_case = float('inf')
        
        for guess in range(start, end + 1):
            cost = guess + max(min_cost(start, guess - 1), 
                              min_cost(guess + 1, end))
            min_worst_case = min(min_worst_case, cost)
        
        memo[(start, end)] = min_worst_case
        return min_worst_case
    
    return min_cost(1, n)


def get_money_amount_interval_dp(n):
    """
    INTERVAL DP APPROACH:
    ====================
    Bottom-up DP processing intervals by length.
    
    Time Complexity: O(n^3) - three nested loops
    Space Complexity: O(n^2) - DP table
    """
    # dp[i][j] = minimum cost to guarantee win in range [i, j]
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Process intervals by length
    for length in range(2, n + 1):  # Need at least 2 numbers for guessing
        for start in range(1, n - length + 2):
            end = start + length - 1
            dp[start][end] = float('inf')
            
            # Try each possible guess in the range
            for guess in range(start, end + 1):
                # Cost of this guess plus worst case of remaining ranges
                left_cost = dp[start][guess - 1] if guess > start else 0
                right_cost = dp[guess + 1][end] if guess < end else 0
                
                cost = guess + max(left_cost, right_cost)
                dp[start][end] = min(dp[start][end], cost)
    
    return dp[1][n]


def get_money_amount_optimized(n):
    """
    OPTIMIZED APPROACH:
    ==================
    Use mathematical insights to reduce constant factors.
    
    Time Complexity: O(n^3) - same asymptotic complexity
    Space Complexity: O(n^2) - DP table
    """
    if n == 1:
        return 0
    
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    for gap in range(2, n + 1):  # gap = end - start + 1
        for start in range(1, n - gap + 2):
            end = start + gap - 1
            dp[start][end] = float('inf')
            
            # Optimization: don't try boundary values for larger ranges
            guess_start = start if gap <= 3 else start + 1
            guess_end = end if gap <= 3 else end - 1
            
            for guess in range(guess_start, guess_end + 1):
                left_cost = dp[start][guess - 1] if guess > start else 0
                right_cost = dp[guess + 1][end] if guess < end else 0
                
                cost = guess + max(left_cost, right_cost)
                dp[start][end] = min(dp[start][end], cost)
    
    return dp[1][n]


def get_money_amount_with_strategy(n):
    """
    TRACK OPTIMAL STRATEGY:
    ======================
    Return minimum cost and the optimal guessing strategy.
    
    Time Complexity: O(n^3) - DP computation + strategy reconstruction
    Space Complexity: O(n^2) - DP table + strategy tracking
    """
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    choice = [[0] * (n + 1) for _ in range(n + 1)]  # Track optimal guesses
    
    for length in range(2, n + 1):
        for start in range(1, n - length + 2):
            end = start + length - 1
            dp[start][end] = float('inf')
            
            for guess in range(start, end + 1):
                left_cost = dp[start][guess - 1] if guess > start else 0
                right_cost = dp[guess + 1][end] if guess < end else 0
                
                cost = guess + max(left_cost, right_cost)
                if cost < dp[start][end]:
                    dp[start][end] = cost
                    choice[start][end] = guess
    
    # Reconstruct strategy
    def build_strategy(start, end):
        if start >= end:
            return None
        
        guess = choice[start][end]
        return {
            'guess': guess,
            'cost': guess,
            'range': (start, end),
            'if_lower': build_strategy(start, guess - 1),
            'if_higher': build_strategy(guess + 1, end)
        }
    
    strategy = build_strategy(1, n)
    return dp[1][n], strategy


def get_money_amount_analysis(n):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and strategy analysis.
    """
    print(f"Guess Number Higher or Lower II Analysis:")
    print(f"Range: 1 to {n}")
    
    # Build DP table with detailed logging
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    choice = [[0] * (n + 1) for _ in range(n + 1)]
    
    print(f"\nDP Table Construction:")
    print(f"dp[i][j] = minimum cost to guarantee win in range [i, j]")
    
    for length in range(2, min(n + 1, 8)):  # Show first few lengths for readability
        print(f"\nLength {length} intervals:")
        for start in range(1, min(n - length + 2, 6)):
            end = start + length - 1
            if end > n:
                continue
                
            dp[start][end] = float('inf')
            
            print(f"  Range [{start}, {end}]:")
            
            best_guess = start
            best_cost = float('inf')
            
            for guess in range(start, end + 1):
                left_cost = dp[start][guess - 1] if guess > start else 0
                right_cost = dp[guess + 1][end] if guess < end else 0
                
                cost = guess + max(left_cost, right_cost)
                
                print(f"    Guess {guess}: cost = {guess} + max({left_cost}, {right_cost}) = {cost}")
                
                if cost < best_cost:
                    best_cost = cost
                    best_guess = guess
                    dp[start][end] = cost
                    choice[start][end] = guess
            
            print(f"    Best: guess {best_guess} with cost {best_cost}")
    
    print(f"\nFinal DP Table (first 8x8):")
    print("   ", end="")
    for j in range(1, min(n + 1, 9)):
        print(f"{j:4}", end="")
    print()
    
    for i in range(1, min(n + 1, 9)):
        print(f"{i:2}: ", end="")
        for j in range(1, min(n + 1, 9)):
            if j >= i:
                print(f"{dp[i][j]:4}", end="")
            else:
                print(f"{'':4}", end="")
        print()
    
    result = dp[1][n] if n > 1 else 0
    print(f"\nMinimum guaranteed cost: {result}")
    
    # Show optimal strategy
    if n <= 10:
        min_cost, strategy = get_money_amount_with_strategy(n)
        print(f"\nOptimal strategy:")
        
        def print_strategy(node, depth=0):
            if node is None:
                return
            
            indent = "  " * depth
            print(f"{indent}Guess {node['guess']} (range {node['range']}, cost: {node['cost']})")
            
            if node['if_lower']:
                print(f"{indent}  If lower:")
                print_strategy(node['if_lower'], depth + 2)
            
            if node['if_higher']:
                print(f"{indent}  If higher:")
                print_strategy(node['if_higher'], depth + 2)
        
        print_strategy(strategy)
        
        # Calculate worst-case path
        def worst_case_path(node):
            if node is None:
                return [], 0
            
            left_path, left_cost = worst_case_path(node['if_lower'])
            right_path, right_cost = worst_case_path(node['if_higher'])
            
            if left_cost >= right_cost:
                return [node['guess']] + left_path, node['cost'] + left_cost
            else:
                return [node['guess']] + right_path, node['cost'] + right_cost
        
        path, total_cost = worst_case_path(strategy)
        print(f"\nWorst-case path: {path}")
        print(f"Total cost: {total_cost}")
    
    return result


def get_money_amount_variants():
    """
    GUESSING GAME VARIANTS:
    ======================
    Different scenarios and modifications.
    """
    
    def get_money_amount_with_limit(n, max_guesses):
        """Limited number of guesses allowed"""
        # This is a more complex variant requiring 3D DP
        # For demonstration, return basic result
        return get_money_amount_interval_dp(n)
    
    def get_money_amount_weighted(n, weights):
        """Different costs for different guesses"""
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        for length in range(2, n + 1):
            for start in range(1, n - length + 2):
                end = start + length - 1
                dp[start][end] = float('inf')
                
                for guess in range(start, end + 1):
                    left_cost = dp[start][guess - 1] if guess > start else 0
                    right_cost = dp[guess + 1][end] if guess < end else 0
                    
                    # Use custom weight instead of guess value
                    cost = weights[guess - 1] + max(left_cost, right_cost)
                    dp[start][end] = min(dp[start][end], cost)
        
        return dp[1][n]
    
    def expected_cost_random_target(n):
        """Expected cost if target is chosen uniformly at random"""
        # This would require a different analysis
        # For now, return the guaranteed worst-case cost
        return get_money_amount_interval_dp(n)
    
    def optimal_first_guess(n):
        """Find the optimal first guess"""
        if n <= 1:
            return 1
        
        min_cost = float('inf')
        best_guess = 1
        
        for guess in range(1, n + 1):
            # Cost if we guess 'guess' first
            left_cost = get_money_amount_interval_dp(guess - 1) if guess > 1 else 0
            right_cost = get_money_amount_interval_dp(n - guess) if guess < n else 0
            
            # Map right_cost to actual range [guess+1, n]
            if guess < n:
                right_cost = get_money_amount_memoization_range(guess + 1, n)
            
            cost = guess + max(left_cost, right_cost)
            
            if cost < min_cost:
                min_cost = cost
                best_guess = guess
        
        return best_guess
    
    def get_money_amount_memoization_range(start, end):
        """Helper for arbitrary range"""
        if start >= end:
            return 0
        
        memo = {}
        
        def min_cost(s, e):
            if s >= e:
                return 0
            
            if (s, e) in memo:
                return memo[(s, e)]
            
            min_worst_case = float('inf')
            
            for guess in range(s, e + 1):
                cost = guess + max(min_cost(s, guess - 1), 
                                  min_cost(guess + 1, e))
                min_worst_case = min(min_worst_case, cost)
            
            memo[(s, e)] = min_worst_case
            return min_worst_case
        
        return min_cost(start, end)
    
    # Test variants
    test_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    print("Guessing Game Variants:")
    print("=" * 50)
    
    for n in test_values:
        print(f"\nn = {n}")
        
        basic_cost = get_money_amount_interval_dp(n)
        print(f"Basic guaranteed cost: {basic_cost}")
        
        if n > 1:
            best_first = optimal_first_guess(n)
            print(f"Optimal first guess: {best_first}")
        
        # With weighted costs (e.g., higher numbers cost more)
        if n <= 10:
            weights = [i * 1.5 for i in range(1, n + 1)]
            weighted_cost = get_money_amount_weighted(n, weights)
            print(f"With 1.5x weights: {weighted_cost:.1f}")


# Test cases
def test_get_money_amount():
    """Test all implementations with various inputs"""
    test_cases = [
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 4),
        (5, 6),
        (6, 8),
        (7, 10),
        (8, 12),
        (9, 14),
        (10, 16)
    ]
    
    print("Testing Guess Number Higher or Lower II Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}")
        print(f"Expected: {expected}")
        
        # Skip recursive for large inputs
        if n <= 7:
            try:
                recursive = get_money_amount_recursive(n)
                print(f"Recursive:        {recursive:>4} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = get_money_amount_memoization(n)
        interval_dp = get_money_amount_interval_dp(n)
        optimized = get_money_amount_optimized(n)
        
        print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        print(f"Interval DP:      {interval_dp:>4} {'✓' if interval_dp == expected else '✗'}")
        print(f"Optimized:        {optimized:>4} {'✓' if optimized == expected else '✗'}")
        
        # Show strategy for small cases
        if n <= 6:
            min_cost, strategy = get_money_amount_with_strategy(n)
            if strategy:
                print(f"First guess in optimal strategy: {strategy['guess']}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    get_money_amount_analysis(6)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    get_money_amount_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MINIMAX STRATEGY: Minimize the maximum possible cost")
    print("2. INTERVAL DP: Optimal strategy for each range")
    print("3. WORST-CASE ANALYSIS: Guarantee win regardless of target")
    print("4. GAME THEORY: Two-player adversarial optimization")
    print("5. BINARY SEARCH LIKE: Optimal guesses tend toward center")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Game Theory: Optimal strategies in adversarial games")
    print("• Decision Making: Worst-case scenario planning")
    print("• Algorithm Design: Minimax optimization problems")
    print("• Search Algorithms: Optimal query strategies")
    print("• Risk Management: Guarantee performance bounds")


if __name__ == "__main__":
    test_get_money_amount()


"""
GUESS NUMBER HIGHER OR LOWER II - MINIMAX GAME THEORY:
======================================================

This problem combines interval DP with game theory:
- Must guarantee a win regardless of opponent's choice
- Minimize the maximum possible cost (minimax strategy)
- Each guess costs its face value
- Opponent provides binary feedback (higher/lower)
- Demonstrates adversarial optimization with interval structure

KEY INSIGHTS:
============
1. **MINIMAX STRATEGY**: Minimize the maximum possible cost
2. **ADVERSARIAL SETTING**: Opponent chooses worst case for us
3. **INTERVAL DECOMPOSITION**: Each guess splits range into two subproblems
4. **GUARANTEED WIN**: Must work for any possible target number
5. **WORST-CASE ANALYSIS**: Take maximum of left and right branches

ALGORITHM APPROACHES:
====================

1. **Recursive (Brute Force)**: O(n!) time, O(n) space
   - Try all possible guessing strategies
   - Exponential branching factor

2. **Memoization**: O(n³) time, O(n²) space
   - Top-down DP with interval caching
   - Natural recursive structure

3. **Interval DP**: O(n³) time, O(n²) space
   - Bottom-up construction by interval length
   - Standard approach for this problem

4. **Optimized DP**: O(n³) time, O(n²) space
   - Same complexity with better constants
   - Practical optimizations

CORE MINIMAX ALGORITHM:
======================
```python
# dp[i][j] = minimum cost to guarantee win in range [i, j]
dp = [[0] * (n+1) for _ in range(n+1)]

for length in range(2, n + 1):
    for start in range(1, n - length + 2):
        end = start + length - 1
        dp[start][end] = inf
        
        for guess in range(start, end + 1):
            left_cost = dp[start][guess-1]   # If target is lower
            right_cost = dp[guess+1][end]    # If target is higher
            
            # Adversary chooses worst case for us
            cost = guess + max(left_cost, right_cost)
            dp[start][end] = min(dp[start][end], cost)
```

RECURRENCE RELATION:
===================
```
dp[i][j] = min(guess + max(dp[i][guess-1], dp[guess+1][j]))
           for all guess in [i, j]

Base cases:
- dp[i][i] = 0     (single number, no guessing needed)
- dp[i][j] = 0     if i > j (empty range)
```

**Intuition**: For range [i, j]:
- Try each possible first guess
- Adversary will choose the number that maximizes our cost
- Take the guess that minimizes this maximum cost

GAME THEORY ANALYSIS:
====================
**Two-Player Game**:
- **Player 1 (Us)**: Choose guessing strategy to minimize cost
- **Player 2 (Adversary)**: Choose target number to maximize our cost
- **Nash Equilibrium**: Optimal strategy for both players

**Minimax Principle**:
- We play optimally assuming adversary plays optimally
- Each node represents a game state (range of possible numbers)
- Value represents guaranteed cost under optimal play

STRATEGY CONSTRUCTION:
=====================
**Decision Tree**: Each guess creates branching:
```
Range [1, n]
├─ Guess k
   ├─ If "lower" → Subgame [1, k-1]
   └─ If "higher" → Subgame [k+1, n]
```

**Optimal First Guess**: Often near center but not always:
- For n=10: optimal first guess is 7 (not 5)
- Balances the worst-case costs of both branches
- Depends on the specific cost structure

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n³) - each of O(n²) intervals considers O(n) guesses
- **Space**: O(n²) - DP table for all intervals
- **States**: O(n²) - all possible ranges [i, j]
- **Transitions**: O(n) - try each guess in range

MATHEMATICAL PROPERTIES:
========================
**Optimal Substructure**: Optimal strategy contains optimal substrategies
**Overlapping Subproblems**: Same ranges appear in multiple contexts
**Monotonicity**: Larger ranges require at least as much cost
**Symmetry**: Often (but not always) optimal to guess near center

SOLUTION PATTERNS:
=================
**Small Examples**:
- n=1: cost=0 (no guessing needed)
- n=2: cost=1 (guess 1, if higher then 2)
- n=3: cost=2 (guess 2, then either 1 or 3)
- n=4: cost=4 (guess 3, covers worst cases optimally)

**General Pattern**: Cost grows roughly as O(n log n)

STRATEGY RECONSTRUCTION:
=======================
To build the actual decision tree:
```python
def build_strategy(start, end, choice):
    if start >= end:
        return None
    
    guess = choice[start][end]
    return {
        'guess': guess,
        'if_lower': build_strategy(start, guess-1, choice),
        'if_higher': build_strategy(guess+1, end, choice)
    }
```

APPLICATIONS:
============
- **Game Theory**: Optimal strategies in adversarial settings
- **Decision Making**: Worst-case scenario planning
- **Search Algorithms**: Optimal query strategies with costs
- **Risk Management**: Guaranteed performance bounds
- **Algorithm Design**: Minimax optimization problems

RELATED PROBLEMS:
================
- **Binary Search**: Similar structure but different objective
- **Optimal Binary Search Trees**: Weighted version with access probabilities
- **Game Theory DP**: Other adversarial optimization problems
- **Decision Trees**: Optimal classification with costs

VARIANTS:
========
- **Limited Guesses**: Constrain number of guesses allowed
- **Weighted Costs**: Different costs for different numbers
- **Multiple Ranges**: Several disjoint ranges to search
- **Expected Cost**: Minimize expected rather than worst-case cost

EDGE CASES:
==========
- **n=1**: No guessing needed, cost=0
- **Small n**: Can be solved by inspection
- **Large n**: Requires efficient DP implementation
- **Degenerate ranges**: Empty or single-element ranges

OPTIMIZATION TECHNIQUES:
=======================
- **Pruning**: Skip obviously suboptimal guesses
- **Symmetry**: Use symmetry to reduce computation
- **Space optimization**: Reduce memory usage
- **Approximation**: Heuristics for very large n

This problem beautifully demonstrates how interval DP can solve
complex adversarial optimization problems, showing the connection
between game theory and dynamic programming.
"""
