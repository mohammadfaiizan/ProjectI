"""
LeetCode 1510: Stone Game IV
Difficulty: Hard
Category: Game Theory DP - Perfect Square Constraints

PROBLEM DESCRIPTION:
===================
Alice and Bob take turns playing a game, with Alice starting first.

Initially, there are n stones in a pile. On each player's turn, that player makes a move consisting of removing any non-zero perfect square number of stones in the pile.

Also, if a player cannot make a move, they lose the game.

Given a positive integer n, return true if and only if Alice wins the game assuming both players play optimally.

Example 1:
Input: n = 1
Output: true
Explanation: Alice can remove 1 stone (which is 1^2) and win, since Bob cannot make any moves.

Example 2:
Input: n = 2
Output: false
Explanation: Alice can only remove 1 stone, after which Bob removes the last one winning the game (2 -> 1 -> 0).

Example 3:
Input: n = 4
Output: true
Explanation: n = 4 -> 2 -> 1 -> 0
Alice removes 2 stones, Bob removes 1 stone, Alice removes the last stone and wins (Alice removes 4 stones total: 2 + 2).

Constraints:
- 1 <= n <= 100000
"""


def stone_game_iv_recursive(n):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to explore all possible perfect square moves.
    
    Time Complexity: O(√n^√n) - exponential explosion
    Space Complexity: O(√n) - recursion stack depth
    """
    def can_win(stones):
        """Return True if current player can win from 'stones' position"""
        if stones == 0:
            return False  # No moves available, current player loses
        
        # Try all perfect square moves
        i = 1
        while i * i <= stones:
            perfect_square = i * i
            # If opponent can't win after this move, current player wins
            if not can_win(stones - perfect_square):
                return True
            i += 1
        
        return False  # No winning move found
    
    return can_win(n)


def stone_game_iv_memoization(n):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed game states.
    
    Time Complexity: O(n * √n) - n states, √n moves per state
    Space Complexity: O(n) - memoization table
    """
    memo = {}
    
    def can_win(stones):
        if stones == 0:
            return False
        
        if stones in memo:
            return memo[stones]
        
        # Try all perfect square moves
        i = 1
        while i * i <= stones:
            perfect_square = i * i
            if not can_win(stones - perfect_square):
                memo[stones] = True
                return True
            i += 1
        
        memo[stones] = False
        return False
    
    return can_win(n)


def stone_game_iv_dp_bottom_up(n):
    """
    BOTTOM-UP DP APPROACH:
    =====================
    Build solutions from 0 to n iteratively.
    
    Time Complexity: O(n * √n) - for each number, try all perfect squares
    Space Complexity: O(n) - DP array
    """
    if n == 0:
        return False
    
    # dp[i] = True if current player can win with i stones
    dp = [False] * (n + 1)
    
    for stones in range(1, n + 1):
        # Try all perfect square moves
        i = 1
        while i * i <= stones:
            perfect_square = i * i
            # If opponent can't win after this move, current player wins
            if not dp[stones - perfect_square]:
                dp[stones] = True
                break
            i += 1
    
    return dp[n]


def stone_game_iv_optimized(n):
    """
    OPTIMIZED DP WITH PRECOMPUTED PERFECT SQUARES:
    ==============================================
    Precompute perfect squares for efficiency.
    
    Time Complexity: O(n * √n) - same complexity, better constants
    Space Complexity: O(n + √n) - DP array + perfect squares list
    """
    if n == 0:
        return False
    
    # Precompute all perfect squares up to n
    perfect_squares = []
    i = 1
    while i * i <= n:
        perfect_squares.append(i * i)
        i += 1
    
    dp = [False] * (n + 1)
    
    for stones in range(1, n + 1):
        for square in perfect_squares:
            if square > stones:
                break
            if not dp[stones - square]:
                dp[stones] = True
                break
    
    return dp[n]


def stone_game_iv_with_analysis(n):
    """
    STONE GAME IV WITH DETAILED ANALYSIS:
    ====================================
    Solve the game and provide comprehensive strategic insights.
    
    Time Complexity: O(n * √n) - DP computation + analysis
    Space Complexity: O(n) - DP table + analysis data
    """
    analysis = {
        'n': n,
        'perfect_squares': [],
        'winning_positions': [],
        'losing_positions': [],
        'optimal_moves': {},
        'pattern_analysis': {},
        'strategy_insights': []
    }
    
    # Precompute perfect squares
    i = 1
    while i * i <= n:
        analysis['perfect_squares'].append(i * i)
        i += 1
    
    # DP with move tracking
    dp = [False] * (n + 1)
    best_moves = [0] * (n + 1)
    
    for stones in range(1, n + 1):
        winning_move = None
        
        for square in analysis['perfect_squares']:
            if square > stones:
                break
            if not dp[stones - square]:
                dp[stones] = True
                winning_move = square
                break
        
        best_moves[stones] = winning_move if winning_move else 0
        
        if dp[stones]:
            analysis['winning_positions'].append(stones)
        else:
            analysis['losing_positions'].append(stones)
    
    analysis['optimal_moves'] = {i: best_moves[i] for i in range(1, n + 1) if best_moves[i] > 0}
    
    # Pattern analysis
    winning_pattern = dp[1:n+1]  # Boolean pattern for positions 1 to n
    
    # Look for patterns in winning/losing positions
    win_lengths = []  # Consecutive winning streaks
    lose_lengths = []  # Consecutive losing streaks
    
    current_win_streak = 0
    current_lose_streak = 0
    
    for i in range(1, n + 1):
        if dp[i]:  # Winning position
            if current_lose_streak > 0:
                lose_lengths.append(current_lose_streak)
                current_lose_streak = 0
            current_win_streak += 1
        else:  # Losing position
            if current_win_streak > 0:
                win_lengths.append(current_win_streak)
                current_win_streak = 0
            current_lose_streak += 1
    
    # Add final streak
    if current_win_streak > 0:
        win_lengths.append(current_win_streak)
    if current_lose_streak > 0:
        lose_lengths.append(current_lose_streak)
    
    analysis['pattern_analysis'] = {
        'winning_pattern': winning_pattern,
        'winning_streaks': win_lengths,
        'losing_streaks': lose_lengths,
        'total_winning': len(analysis['winning_positions']),
        'total_losing': len(analysis['losing_positions']),
        'win_percentage': len(analysis['winning_positions']) / n * 100
    }
    
    # Strategic insights
    if dp[n]:
        analysis['strategy_insights'].append(f"Alice wins with optimal first move: {best_moves[n]}")
    else:
        analysis['strategy_insights'].append("Alice has no winning strategy")
    
    analysis['strategy_insights'].append(f"Available perfect squares: {analysis['perfect_squares']}")
    analysis['strategy_insights'].append(f"Total winning positions: {len(analysis['winning_positions'])}/{n}")
    
    # Analyze critical positions
    large_squares = [sq for sq in analysis['perfect_squares'] if sq >= n // 2]
    if large_squares:
        analysis['strategy_insights'].append(f"Large moves available: {large_squares}")
    
    return dp[n], analysis


def stone_game_iv_analysis(n):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze Stone Game IV with detailed strategic insights.
    """
    print(f"Stone Game IV Analysis for n = {n}:")
    print(f"Starting stones: {n}")
    
    # Find available perfect squares
    perfect_squares = []
    i = 1
    while i * i <= n:
        perfect_squares.append(i * i)
        i += 1
    
    print(f"Available perfect square moves: {perfect_squares}")
    print(f"Number of move options: {len(perfect_squares)}")
    
    # Different approaches
    if n <= 50:
        try:
            recursive = stone_game_iv_recursive(n)
            print(f"Recursive result: {recursive}")
        except:
            print("Recursive: Too slow")
    
    memoization = stone_game_iv_memoization(n)
    bottom_up = stone_game_iv_dp_bottom_up(n)
    optimized = stone_game_iv_optimized(n)
    
    print(f"Memoization result: {memoization}")
    print(f"Bottom-up DP result: {bottom_up}")
    print(f"Optimized result: {optimized}")
    
    # Detailed analysis
    detailed_result, analysis = stone_game_iv_with_analysis(n)
    
    print(f"\nDetailed Game Analysis:")
    print(f"Alice can win: {detailed_result}")
    
    if analysis['optimal_moves'].get(n):
        print(f"Optimal first move: Remove {analysis['optimal_moves'][n]} stones")
    
    print(f"\nPosition Analysis:")
    pattern = analysis['pattern_analysis']
    print(f"Winning positions: {pattern['total_winning']}/{n} ({pattern['win_percentage']:.1f}%)")
    print(f"Losing positions: {pattern['total_losing']}/{n}")
    
    if n <= 20:
        print(f"\nPosition-by-position breakdown:")
        for i in range(1, min(n + 1, 21)):
            status = "WIN" if i in analysis['winning_positions'] else "LOSE"
            move = analysis['optimal_moves'].get(i, 0)
            move_str = f"(remove {move})" if move > 0 else ""
            print(f"  {i:2d} stones: {status:4s} {move_str}")
    
    print(f"\nPattern Analysis:")
    if pattern['winning_streaks']:
        print(f"Winning streaks: {pattern['winning_streaks']}")
    if pattern['losing_streaks']:
        print(f"Losing streaks: {pattern['losing_streaks']}")
    
    print(f"\nStrategy Insights:")
    for insight in analysis['strategy_insights']:
        print(f"  • {insight}")
    
    # Mathematical properties
    print(f"\nMathematical Properties:")
    print(f"  • Perfect squares up to {n}: {len(perfect_squares)}")
    print(f"  • Largest perfect square ≤ {n}: {perfect_squares[-1] if perfect_squares else 0}")
    print(f"  • Game always terminates (stones decrease)")
    print(f"  • No draws possible (finite game tree)")
    
    return detailed_result


def stone_game_iv_variants():
    """
    STONE GAME IV VARIANTS:
    ======================
    Different rule modifications and extensions.
    """
    
    def stone_game_perfect_cubes(n):
        """Stone game with perfect cubes instead of squares"""
        memo = {}
        
        def can_win(stones):
            if stones == 0:
                return False
            
            if stones in memo:
                return memo[stones]
            
            # Try all perfect cube moves
            i = 1
            while i * i * i <= stones:
                perfect_cube = i * i * i
                if not can_win(stones - perfect_cube):
                    memo[stones] = True
                    return True
                i += 1
            
            memo[stones] = False
            return False
        
        return can_win(n)
    
    def stone_game_fibonacci_numbers(n):
        """Stone game with Fibonacci numbers"""
        # Generate Fibonacci numbers up to n
        fibs = [1, 1]
        while fibs[-1] < n:
            fibs.append(fibs[-1] + fibs[-2])
        fibs = [f for f in fibs if f <= n]
        
        memo = {}
        
        def can_win(stones):
            if stones == 0:
                return False
            
            if stones in memo:
                return memo[stones]
            
            for fib in fibs:
                if fib > stones:
                    break
                if not can_win(stones - fib):
                    memo[stones] = True
                    return True
            
            memo[stones] = False
            return False
        
        return can_win(n)
    
    def stone_game_prime_numbers(n):
        """Stone game with prime numbers"""
        # Generate primes up to n using simple sieve
        def sieve(limit):
            is_prime = [True] * (limit + 1)
            is_prime[0] = is_prime[1] = False
            
            for i in range(2, int(limit**0.5) + 1):
                if is_prime[i]:
                    for j in range(i*i, limit + 1, i):
                        is_prime[j] = False
            
            return [i for i in range(2, limit + 1) if is_prime[i]]
        
        primes = sieve(n)
        memo = {}
        
        def can_win(stones):
            if stones == 0:
                return False
            
            if stones in memo:
                return memo[stones]
            
            for prime in primes:
                if prime > stones:
                    break
                if not can_win(stones - prime):
                    memo[stones] = True
                    return True
            
            memo[stones] = False
            return False
        
        return can_win(n)
    
    def stone_game_powers_of_two(n):
        """Stone game with powers of 2"""
        memo = {}
        
        def can_win(stones):
            if stones == 0:
                return False
            
            if stones in memo:
                return memo[stones]
            
            # Try all powers of 2
            power = 1
            while power <= stones:
                if not can_win(stones - power):
                    memo[stones] = True
                    return True
                power *= 2
            
            memo[stones] = False
            return False
        
        return can_win(n)
    
    # Test variants
    test_values = [1, 2, 3, 4, 5, 8, 10, 15, 20]
    
    print("Stone Game IV Variants:")
    print("=" * 50)
    
    for n in test_values:
        print(f"\nn = {n}:")
        
        basic_result = stone_game_iv_optimized(n)
        print(f"Perfect squares: Alice wins: {basic_result}")
        
        # Perfect cubes variant
        cubes_result = stone_game_perfect_cubes(n)
        print(f"Perfect cubes: Alice wins: {cubes_result}")
        
        # Fibonacci variant
        if n <= 25:  # Limit for performance
            fib_result = stone_game_fibonacci_numbers(n)
            print(f"Fibonacci numbers: Alice wins: {fib_result}")
        
        # Prime numbers variant
        if n <= 20:  # Limit for performance
            prime_result = stone_game_prime_numbers(n)
            print(f"Prime numbers: Alice wins: {prime_result}")
        
        # Powers of 2 variant
        powers_result = stone_game_powers_of_two(n)
        print(f"Powers of 2: Alice wins: {powers_result}")


# Test cases
def test_stone_game_iv():
    """Test all implementations with various inputs"""
    test_cases = [
        (1, True),
        (2, False),
        (3, True),
        (4, True),
        (5, False),
        (6, True),
        (7, False),
        (8, True),
        (9, True),
        (10, True)
    ]
    
    print("Testing Stone Game IV Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}")
        print(f"Expected: {expected}")
        
        # Skip recursive for larger cases
        if n <= 20:
            try:
                recursive = stone_game_iv_recursive(n)
                print(f"Recursive:        {recursive} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = stone_game_iv_memoization(n)
        bottom_up = stone_game_iv_dp_bottom_up(n)
        optimized = stone_game_iv_optimized(n)
        
        print(f"Memoization:      {memoization} {'✓' if memoization == expected else '✗'}")
        print(f"Bottom-up DP:     {bottom_up} {'✓' if bottom_up == expected else '✗'}")
        print(f"Optimized:        {optimized} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    stone_game_iv_analysis(20)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    stone_game_iv_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. PERFECT SQUARE CONSTRAINTS: Limited move options based on mathematical property")
    print("2. WINNING PATTERN: Complex pattern not easily predictable")
    print("3. MOVE EFFICIENCY: Larger moves not always better due to opponent response")
    print("4. STATE SPACE: Linear in n, but transitions depend on square root")
    print("5. MATHEMATICAL STRUCTURE: Number theory influences game strategy")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Mathematical Games: Games with number-theoretic constraints")
    print("• Algorithm Design: DP with mathematical constraint sets")
    print("• Competitive Programming: Number theory in game theory problems")
    print("• Educational: Teaching DP with interesting constraint patterns")
    print("• Pattern Recognition: Analyzing winning/losing position patterns")


if __name__ == "__main__":
    test_stone_game_iv()


"""
STONE GAME IV - GAME THEORY WITH PERFECT SQUARE CONSTRAINTS:
============================================================

This problem introduces mathematical constraints to game theory DP:
- Move options limited to perfect squares (1, 4, 9, 16, ...)
- Number theory influences strategic possibilities
- Complex winning/losing patterns not easily predictable
- Demonstrates how mathematical constraints affect game dynamics

KEY INSIGHTS:
============
1. **PERFECT SQUARE CONSTRAINTS**: Move options restricted by mathematical property
2. **IRREGULAR PATTERNS**: Winning positions don't follow simple patterns
3. **MATHEMATICAL INFLUENCE**: Number theory directly affects game strategy
4. **CONSTRAINT-DRIVEN STRATEGY**: Available moves depend on current stone count
5. **NON-MONOTONIC DIFFICULTY**: Larger n doesn't necessarily favor first player

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(√n^√n) time, O(√n) space
   - Pure recursive exploration of perfect square moves
   - Exponential complexity without memoization

2. **Memoization**: O(n√n) time, O(n) space
   - Top-down DP with state caching
   - Each position computed once with √n move evaluations

3. **Bottom-up DP**: O(n√n) time, O(n) space
   - Iterative DP building from smaller positions
   - Most intuitive for understanding position dependencies

4. **Optimized DP**: O(n√n) time, O(n + √n) space
   - Precompute perfect squares for efficiency
   - Better constant factors in practice

CORE CONSTRAINT-BASED ALGORITHM:
================================
```python
def winnerSquareGame(n):
    # dp[i] = True if current player can win with i stones
    dp = [False] * (n + 1)
    
    for stones in range(1, n + 1):
        # Try all perfect square moves
        i = 1
        while i * i <= stones:
            square = i * i
            # If opponent loses after this move, current player wins
            if not dp[stones - square]:
                dp[stones] = True
                break
            i += 1
    
    return dp[n]
```

PERFECT SQUARE GENERATION:
==========================
**Move Set**: {1², 2², 3², ..., k²} where k² ≤ current_stones
**Dynamic Availability**: Move options change based on current position
**Strategic Implications**: Large jumps possible but may benefit opponent

WINNING PATTERN ANALYSIS:
========================
**Complex Patterns**: Unlike simple parity games, winning positions show irregular patterns
**No Simple Formula**: Cannot predict winner from n alone without computation
**Position Dependencies**: Winning status depends on intricate interaction of available moves

**Pattern Characteristics**:
- Positions 1, 3, 4, 6, 8, 9, 10... are often winning for first player
- Positions 2, 5, 7... are often losing for first player
- Pattern becomes more complex as n increases

MATHEMATICAL CONSTRAINTS IMPACT:
===============================
**Move Limitation**: Not all decrements possible, only perfect squares
**Strategic Depth**: Must consider which perfect squares to use
**Endgame Analysis**: Small positions have limited move options

**Number Theory Influence**:
- Dense vs sparse perfect square availability
- Large gaps between consecutive perfect squares
- Strategic value of small vs large moves

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n√n)
- n positions to compute
- Up to √n perfect square moves per position
- Total: O(n × √n) = O(n^1.5)

**Space Complexity**: O(n) for DP array, O(√n) for perfect squares list

**Practical Performance**: Efficient for moderate n (≤ 10⁵)

STRATEGIC CONSIDERATIONS:
========================
**Move Size Trade-offs**:
- Small moves (1): Always available, conservative play
- Medium moves (4, 9, 16): Balance between progress and risk
- Large moves: High impact but limited availability

**Opponent Response**: Must consider how each move affects opponent's options

**Endgame Planning**: Final moves often forced by remaining stone count

CONSTRAINT PATTERN ANALYSIS:
===========================
**Perfect Square Density**: √n moves available for position n
**Gap Analysis**: Larger gaps between consecutive squares as numbers increase
**Critical Positions**: Positions just above perfect squares often strategic

APPLICATIONS:
============
- **Mathematical Games**: Games with number-theoretic move constraints
- **Constraint Programming**: Optimization with mathematical restrictions
- **Educational Tools**: Teaching DP with interesting mathematical patterns
- **Algorithm Design**: Handling irregular constraint patterns
- **Pattern Recognition**: Analyzing complex winning/losing sequences

RELATED PROBLEMS:
================
- **Stone Game Variants**: Different move constraint patterns
- **Nim Games**: Classical combinatorial game theory
- **Number Theory Games**: Games based on mathematical properties
- **Constraint Games**: Games with specific move limitations

VARIANTS:
========
- **Perfect Cubes**: Use 1³, 2³, 3³, ... instead of squares
- **Fibonacci Numbers**: Moves limited to Fibonacci sequence
- **Prime Numbers**: Only prime number decrements allowed
- **Powers of 2**: Moves limited to 1, 2, 4, 8, 16, ...

EDGE CASES:
==========
- **n = 1**: Only move is 1², Alice wins immediately
- **n = 2**: Alice takes 1, Bob takes 1, Alice loses
- **Perfect Squares**: When n is perfect square, often strategic
- **Large n**: Pattern complexity increases significantly

OPTIMIZATION TECHNIQUES:
=======================
**Perfect Square Precomputation**: Calculate once, reuse
**Early Termination**: Stop when first winning move found
**Pattern Recognition**: Look for mathematical patterns in solutions
**Memory Optimization**: Space-efficient DP implementation

This problem showcases how mathematical constraints can create
rich strategic depth in game theory, demonstrating the interplay
between number theory and optimal decision making while
highlighting the complexity that arises from seemingly simple
mathematical restrictions on game moves.
"""
