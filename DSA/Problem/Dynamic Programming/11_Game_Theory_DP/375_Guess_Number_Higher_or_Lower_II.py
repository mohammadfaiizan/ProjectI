"""
LeetCode 375: Guess Number Higher or Lower II
Difficulty: Medium
Category: Game Theory DP - Minimax Cost Optimization

PROBLEM DESCRIPTION:
===================
We are playing the Guessing Game. The game will work as follows:

1. I pick a number between 1 and n.
2. You guess a number.
3. If you guess the right number, you win the game.
4. If you guess the wrong number, then I will tell you whether the number I picked is higher or lower than your guess, and you have to continue guessing.
5. Every time you guess a wrong number x, you have to pay $x. If you guess the number, you don't have to pay anything.

What is the amount of money you need to guarantee that you can always find the number, regardless of which number I pick?

Example 1:
Input: n = 10
Output: 16
Explanation: The winning strategy is:
- 1st guess: 7, cost = 7
- If the number is 1-6, 2nd guess: 3, cost = 3
- If the number is 8-10, 2nd guess: 9, cost = 9
- Continue this strategy to guarantee finding the number.
The worst case cost is 16.

Example 2:
Input: n = 1
Output: 0

Example 3:
Input: n = 2
Output: 1

Constraints:
- 1 <= n <= 200
"""


def guess_number_higher_lower_recursive(n):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to find minimum cost strategy.
    
    Time Complexity: O(n!) - exponential without memoization
    Space Complexity: O(n) - recursion stack
    """
    def min_cost(start, end):
        """
        Return minimum cost to guarantee finding number in range [start, end]
        """
        if start >= end:
            return 0
        
        min_worst_cost = float('inf')
        
        # Try each possible guess
        for guess in range(start, end + 1):
            # Cost for this guess + worst case cost for both sides
            left_cost = min_cost(start, guess - 1)  # Number is lower
            right_cost = min_cost(guess + 1, end)   # Number is higher
            
            # Worst case: we pay guess + max of left or right subtree cost
            worst_case_cost = guess + max(left_cost, right_cost)
            min_worst_cost = min(min_worst_cost, worst_case_cost)
        
        return min_worst_cost
    
    return min_cost(1, n)


def guess_number_higher_lower_memoization(n):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed ranges.
    
    Time Complexity: O(n^3) - n^2 states, n choices per state
    Space Complexity: O(n^2) - memoization table
    """
    memo = {}
    
    def min_cost(start, end):
        if start >= end:
            return 0
        
        if (start, end) in memo:
            return memo[(start, end)]
        
        min_worst_cost = float('inf')
        
        for guess in range(start, end + 1):
            left_cost = min_cost(start, guess - 1)
            right_cost = min_cost(guess + 1, end)
            worst_case_cost = guess + max(left_cost, right_cost)
            min_worst_cost = min(min_worst_cost, worst_case_cost)
        
        memo[(start, end)] = min_worst_cost
        return min_worst_cost
    
    return min_cost(1, n)


def guess_number_higher_lower_dp(n):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Use bottom-up DP with interval building.
    
    Time Complexity: O(n^3) - three nested loops
    Space Complexity: O(n^2) - DP table
    """
    # dp[i][j] = minimum cost to guarantee finding number in range [i, j]
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Fill DP table for increasing interval lengths
    for length in range(2, n + 1):  # length of interval
        for start in range(1, n - length + 2):
            end = start + length - 1
            dp[start][end] = float('inf')
            
            # Try each possible guess in this range
            for guess in range(start, end + 1):
                left_cost = dp[start][guess - 1] if guess > start else 0
                right_cost = dp[guess + 1][end] if guess < end else 0
                
                worst_case_cost = guess + max(left_cost, right_cost)
                dp[start][end] = min(dp[start][end], worst_case_cost)
    
    return dp[1][n]


def guess_number_higher_lower_with_strategy(n):
    """
    GUESS NUMBER WITH STRATEGY RECONSTRUCTION:
    =========================================
    Find optimal strategy and reconstruct decision tree.
    
    Time Complexity: O(n^3) - DP computation + strategy tracking
    Space Complexity: O(n^2) - DP table + strategy data
    """
    # DP table and strategy tracking
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    strategy = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Fill DP table with strategy tracking
    for length in range(2, n + 1):
        for start in range(1, n - length + 2):
            end = start + length - 1
            dp[start][end] = float('inf')
            best_guess = start
            
            for guess in range(start, end + 1):
                left_cost = dp[start][guess - 1] if guess > start else 0
                right_cost = dp[guess + 1][end] if guess < end else 0
                
                worst_case_cost = guess + max(left_cost, right_cost)
                
                if worst_case_cost < dp[start][end]:
                    dp[start][end] = worst_case_cost
                    best_guess = guess
            
            strategy[start][end] = best_guess
    
    # Reconstruct decision tree
    def build_strategy_tree(start, end, depth=0):
        if start > end:
            return None
        if start == end:
            return {'guess': start, 'cost': 0, 'depth': depth, 'children': {}}
        
        guess = strategy[start][end]
        cost = dp[start][end]
        
        tree = {
            'guess': guess,
            'cost': cost,
            'depth': depth,
            'children': {}
        }
        
        # Left subtree (number is lower)
        if guess > start:
            tree['children']['lower'] = build_strategy_tree(start, guess - 1, depth + 1)
        
        # Right subtree (number is higher)
        if guess < end:
            tree['children']['higher'] = build_strategy_tree(guess + 1, end, depth + 1)
        
        return tree
    
    strategy_tree = build_strategy_tree(1, n)
    return dp[1][n], strategy_tree


def guess_number_higher_lower_analysis(n):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the guessing game with detailed strategic insights.
    """
    print(f"Guess Number Higher or Lower II Analysis for n = {n}:")
    print(f"Number range: 1 to {n}")
    print(f"Game objective: Minimize worst-case cost to guarantee finding any number")
    
    if n == 1:
        print(f"Trivial case: Only one number, no guessing needed")
        return 0
    
    # Different approaches
    if n <= 10:
        try:
            recursive = guess_number_higher_lower_recursive(n)
            print(f"Recursive result: {recursive}")
        except:
            print("Recursive: Too slow")
    
    memoization = guess_number_higher_lower_memoization(n)
    dp_result = guess_number_higher_lower_dp(n)
    
    print(f"Memoization result: {memoization}")
    print(f"DP result: {dp_result}")
    
    # Strategy analysis
    optimal_cost, strategy_tree = guess_number_higher_lower_with_strategy(n)
    
    print(f"\nOptimal Strategy Analysis:")
    print(f"Minimum guaranteed cost: {optimal_cost}")
    
    def print_strategy(node, range_start, range_end, prefix=""):
        if not node:
            return
        
        guess = node['guess']
        cost = node['cost']
        depth = node['depth']
        
        print(f"{prefix}Range [{range_start}-{range_end}]: Guess {guess} (cost: {cost}, depth: {depth})")
        
        # Print children
        if 'lower' in node['children'] and node['children']['lower']:
            print(f"{prefix}  If number < {guess}:")
            print_strategy(node['children']['lower'], range_start, guess - 1, prefix + "    ")
        
        if 'higher' in node['children'] and node['children']['higher']:
            print(f"{prefix}  If number > {guess}:")
            print_strategy(node['children']['higher'], guess + 1, range_end, prefix + "    ")
    
    print(f"\nOptimal Decision Tree:")
    print_strategy(strategy_tree, 1, n)
    
    # Analyze strategy properties
    def analyze_tree(node):
        if not node:
            return {'max_depth': 0, 'total_nodes': 0, 'guesses': []}
        
        stats = {
            'max_depth': node['depth'],
            'total_nodes': 1,
            'guesses': [node['guess']]
        }
        
        for child in node['children'].values():
            if child:
                child_stats = analyze_tree(child)
                stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'])
                stats['total_nodes'] += child_stats['total_nodes']
                stats['guesses'].extend(child_stats['guesses'])
        
        return stats
    
    tree_stats = analyze_tree(strategy_tree)
    
    print(f"\nStrategy Properties:")
    print(f"Maximum search depth: {tree_stats['max_depth']}")
    print(f"Total decision nodes: {tree_stats['total_nodes']}")
    print(f"First guess: {strategy_tree['guess'] if strategy_tree else 'N/A'}")
    
    # Compare with binary search
    binary_search_depth = (n - 1).bit_length()  # ceiling of log2(n)
    print(f"Binary search would need depth: {binary_search_depth}")
    print(f"Optimal strategy depth: {tree_stats['max_depth']}")
    
    # Cost analysis
    print(f"\nCost Analysis:")
    print(f"If we always guessed randomly, expected cost could be much higher")
    print(f"Optimal strategy guarantees cost ≤ {optimal_cost} regardless of target")
    print(f"Average number per guess: {sum(tree_stats['guesses']) / len(tree_stats['guesses']):.2f}")
    
    return optimal_cost


def guess_number_higher_lower_variants():
    """
    GUESS NUMBER VARIANTS:
    =====================
    Different game rule modifications.
    """
    
    def guess_with_fixed_cost(n, fixed_cost):
        """Variant where every wrong guess costs fixed amount"""
        memo = {}
        
        def min_cost(start, end):
            if start >= end:
                return 0
            
            if (start, end) in memo:
                return memo[(start, end)]
            
            min_worst_cost = float('inf')
            
            for guess in range(start, end + 1):
                left_cost = min_cost(start, guess - 1)
                right_cost = min_cost(guess + 1, end)
                
                # Fixed cost instead of guess value
                worst_case_cost = fixed_cost + max(left_cost, right_cost)
                min_worst_cost = min(min_worst_cost, worst_case_cost)
            
            memo[(start, end)] = min_worst_cost
            return min_worst_cost
        
        return min_cost(1, n)
    
    def guess_with_bonus(n, bonus_threshold):
        """Variant where guessing close to target gives bonus"""
        memo = {}
        
        def min_cost(start, end):
            if start >= end:
                return 0
            
            if (start, end) in memo:
                return memo[(start, end)]
            
            min_worst_cost = float('inf')
            
            for guess in range(start, end + 1):
                left_cost = min_cost(start, guess - 1)
                right_cost = min_cost(guess + 1, end)
                
                # Reduced cost if range is small (close guess)
                cost_multiplier = 1.0
                if end - start + 1 <= bonus_threshold:
                    cost_multiplier = 0.5
                
                worst_case_cost = guess * cost_multiplier + max(left_cost, right_cost)
                min_worst_cost = min(min_worst_cost, worst_case_cost)
            
            memo[(start, end)] = min_worst_cost
            return min_worst_cost
        
        return min_cost(1, n)
    
    def guess_with_limited_guesses(n, max_guesses):
        """Variant with limited number of guesses"""
        memo = {}
        
        def can_guarantee(start, end, guesses_left):
            if start >= end:
                return True
            if guesses_left <= 0:
                return False
            
            state = (start, end, guesses_left)
            if state in memo:
                return memo[state]
            
            # Try each possible guess
            for guess in range(start, end + 1):
                left_ok = can_guarantee(start, guess - 1, guesses_left - 1)
                right_ok = can_guarantee(guess + 1, end, guesses_left - 1)
                
                if left_ok and right_ok:
                    memo[state] = True
                    return True
            
            memo[state] = False
            return False
        
        return can_guarantee(1, n, max_guesses)
    
    def optimal_binary_search_cost(n):
        """Cost if we used optimal binary search strategy"""
        # Binary search always guesses middle
        memo = {}
        
        def binary_cost(start, end):
            if start >= end:
                return 0
            
            if (start, end) in memo:
                return memo[(start, end)]
            
            mid = (start + end) // 2
            left_cost = binary_cost(start, mid - 1)
            right_cost = binary_cost(mid + 1, end)
            
            cost = mid + max(left_cost, right_cost)
            memo[(start, end)] = cost
            return cost
        
        return binary_cost(1, n)
    
    # Test variants
    test_values = [1, 2, 3, 5, 8, 10, 15]
    
    print("Guess Number Game Variants:")
    print("=" * 50)
    
    for n in test_values:
        print(f"\nn = {n}:")
        
        if n <= 1:
            print(f"Basic game: 0 (trivial)")
            continue
        
        basic_cost = guess_number_higher_lower_dp(n)
        print(f"Basic game optimal cost: {basic_cost}")
        
        # Fixed cost variant
        fixed_cost = guess_with_fixed_cost(n, 1)
        print(f"Fixed cost per guess (1): {fixed_cost}")
        
        # Bonus variant
        if n >= 3:
            bonus_cost = guess_with_bonus(n, 3)
            print(f"With bonus for close guesses: {bonus_cost:.1f}")
        
        # Limited guesses
        max_guesses_needed = (n - 1).bit_length() + 2  # Conservative estimate
        can_guarantee = guess_with_limited_guesses(n, max_guesses_needed)
        print(f"Can guarantee with {max_guesses_needed} guesses: {can_guarantee}")
        
        # Binary search comparison
        binary_cost = optimal_binary_search_cost(n)
        print(f"Binary search cost: {binary_cost}")
        print(f"Optimal vs Binary: {basic_cost} vs {binary_cost}")


# Test cases
def test_guess_number_higher_lower():
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
        (10, 16)
    ]
    
    print("Testing Guess Number Higher or Lower II Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}")
        print(f"Expected: {expected}")
        
        # Skip recursive for larger cases
        if n <= 8:
            try:
                recursive = guess_number_higher_lower_recursive(n)
                print(f"Recursive:        {recursive:>4} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = guess_number_higher_lower_memoization(n)
        dp_result = guess_number_higher_lower_dp(n)
        
        print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        print(f"DP:               {dp_result:>4} {'✓' if dp_result == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    guess_number_higher_lower_analysis(10)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    guess_number_higher_lower_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MINIMAX OPTIMIZATION: Minimize worst-case cost across all scenarios")
    print("2. INTERVAL DP: Optimal substructure over number ranges")
    print("3. STRATEGIC GUESSING: Balance risk across left and right subtrees")
    print("4. COST AWARENESS: Guess value matters, not just search efficiency")
    print("5. DECISION TREE: Optimal strategy forms a decision tree")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Game Theory: Optimal strategy under worst-case scenarios")
    print("• Decision Trees: Optimal decision making with costs")
    print("• Risk Management: Minimizing maximum potential loss")
    print("• Algorithm Design: Cost-aware search and optimization")
    print("• Competitive Programming: Minimax problems with cost functions")


if __name__ == "__main__":
    test_guess_number_higher_lower()


"""
GUESS NUMBER HIGHER OR LOWER II - MINIMAX COST OPTIMIZATION:
============================================================

This problem demonstrates minimax optimization with cost considerations:
- Minimize worst-case cost rather than average case
- Strategic decision making balancing multiple risk scenarios
- Interval DP with cost-aware optimal substructure
- Decision tree construction for guaranteed performance bounds

KEY INSIGHTS:
============
1. **MINIMAX OPTIMIZATION**: Minimize the maximum cost across all possible scenarios
2. **COST-AWARE STRATEGY**: Guess value becomes the cost, affecting optimal choices
3. **WORST-CASE GUARANTEE**: Strategy must work regardless of which number is chosen
4. **INTERVAL DP**: Optimal substructure over contiguous number ranges
5. **STRATEGIC BALANCE**: Balance cost between left and right search subtrees

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(n!) time, O(n) space
   - Pure recursive exploration without memoization
   - Exponential complexity due to overlapping subproblems

2. **Memoization**: O(n³) time, O(n²) space
   - Top-down DP with range-based state caching
   - Each (start, end) range computed once

3. **Bottom-up DP**: O(n³) time, O(n²) space
   - Iterative DP building from smaller to larger intervals
   - Most common approach for interval DP problems

4. **Strategy Reconstruction**: O(n³) time, O(n²) space
   - Include optimal decision tracking for strategy tree
   - Essential for understanding optimal play patterns

CORE MINIMAX COST ALGORITHM:
===========================
```python
def getMoneyAmount(n):
    # dp[i][j] = min cost to guarantee finding number in range [i,j]
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Fill for increasing interval lengths
    for length in range(2, n + 1):
        for start in range(1, n - length + 2):
            end = start + length - 1
            dp[start][end] = float('inf')
            
            # Try each possible guess in this range
            for guess in range(start, end + 1):
                left_cost = dp[start][guess - 1] if guess > start else 0
                right_cost = dp[guess + 1][end] if guess < end else 0
                
                # Worst case: pay guess + max of both subtrees
                worst_case = guess + max(left_cost, right_cost)
                dp[start][end] = min(dp[start][end], worst_case)
    
    return dp[1][n]
```

MINIMAX PRINCIPLE:
=================
**Worst-Case Optimization**: For each guess, consider the worst possible outcome
- If target is lower: pay guess + optimal cost for left range
- If target is higher: pay guess + optimal cost for right range
- Worst case: max of these two scenarios

**Strategic Implication**: Choose guess that minimizes maximum possible cost

COST-AWARE DECISION MAKING:
==========================
**Cost Function**: Wrong guess costs the guessed number
- Higher guesses are more expensive when wrong
- Lower guesses are cheaper but may lead to expensive subtrees
- Must balance immediate cost with future cost potential

**Strategic Trade-offs**:
- Conservative low guesses: cheap mistakes, potentially deep search
- Aggressive high guesses: expensive mistakes, potentially quick resolution
- Optimal strategy: mathematical balance of these factors

INTERVAL DP STRUCTURE:
=====================
**State Definition**: `dp[i][j]` = minimum cost to guarantee finding number in range [i,j]

**Transition**: For range [i,j], try each guess k ∈ [i,j]:
```
cost = k + max(dp[i][k-1], dp[k+1][j])
dp[i][j] = min over all k of cost
```

**Base Cases**: 
- `dp[i][i] = 0` (single number, no guessing needed)
- `dp[i][j] = 0` when i > j (empty range)

STRATEGY TREE CONSTRUCTION:
===========================
**Decision Tree**: Optimal strategy forms a binary decision tree
- Each node represents a guess for a range
- Left child: what to do if target is lower
- Right child: what to do if target is higher

**Tree Properties**:
- Depth varies based on worst-case path
- Each path guarantees finding target within cost bound
- Non-uniform branching based on cost optimization

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n³)
- n² possible intervals [i,j]
- n possible guesses per interval
- Total: O(n² × n) = O(n³)

**Space Complexity**: O(n²)
- DP table storage
- Strategy tracking (optional)

**Practical Performance**: Efficient for reasonable n (≤ 200)

COMPARISON WITH BINARY SEARCH:
==============================
**Binary Search**: Always guess middle, optimizes comparison count
**Optimal Strategy**: Considers guess cost, may deviate from middle

**Trade-off Analysis**:
- Binary search: log₂(n) comparisons, variable cost per comparison
- Optimal strategy: potentially more comparisons, minimized total cost

APPLICATIONS:
============
- **Game Theory**: Optimal strategy under worst-case scenarios
- **Risk Management**: Minimizing maximum potential loss
- **Decision Trees**: Cost-aware optimal decision making
- **Search Algorithms**: Cost-sensitive search optimization
- **Competitive Programming**: Minimax problems with cost functions

RELATED PROBLEMS:
================
- **Binary Search Variations**: Different cost models
- **Minimax Games**: Two-player optimization
- **Interval DP**: Range-based optimization problems
- **Decision Theory**: Optimal choice under uncertainty

VARIANTS:
========
- **Fixed Cost per Guess**: Every wrong guess costs same amount
- **Bonus Systems**: Rewards for close guesses
- **Limited Guesses**: Maximum number of attempts allowed
- **Variable Ranges**: Different starting ranges

EDGE CASES:
==========
- **n = 1**: No guessing needed, cost = 0
- **n = 2**: Optimal to guess 1 first, cost = 1
- **Small Ranges**: Often optimal to guess conservatively
- **Large Ranges**: Complex balance between immediate and future costs

OPTIMIZATION TECHNIQUES:
=======================
**Symmetry**: Leverage symmetric properties in strategy
**Pruning**: Early termination when cost bounds exceeded
**Memory Optimization**: Space-efficient DP implementation
**Strategy Caching**: Reuse computed optimal strategies

This problem demonstrates how cost considerations can
significantly alter optimal strategies compared to
traditional search algorithms, requiring sophisticated
analysis to balance immediate costs with future risks
in adversarial or worst-case scenarios.
"""
