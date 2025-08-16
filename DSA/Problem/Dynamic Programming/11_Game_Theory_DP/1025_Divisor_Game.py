"""
LeetCode 1025: Divisor Game
Difficulty: Easy
Category: Game Theory DP - Mathematical Game Theory

PROBLEM DESCRIPTION:
===================
Alice and Bob take turns playing a game, with Alice starting first.

Initially, there is a number n on the chalkboard. On each player's turn, that player makes a move consisting of:
- Choosing any x with 0 < x < n and n % x == 0.
- Replacing the number on the chalkboard with n - x.

Also, if a player cannot make a move, they lose the game.

Return true if and only if Alice has a winning strategy.

Example 1:
Input: n = 2
Output: true
Explanation: Alice chooses 1, Bob receives 1 and can't move.

Example 2:
Input: n = 3
Output: false
Explanation: Alice chooses 1, the only available move, and Bob receives 2.
Bob chooses 1, Alice receives 1 and can't move.

Constraints:
- 1 <= n <= 1000
"""


def divisor_game_mathematical_insight(n):
    """
    MATHEMATICAL INSIGHT:
    ====================
    Alice wins if and only if n is even.
    
    Time Complexity: O(1) - constant time
    Space Complexity: O(1) - no additional space
    """
    # Mathematical proof:
    # - If n is even, Alice can always choose x=1 (since n%1==0), leaving Bob with odd n-1
    # - If n is odd, all divisors of n are odd, so Alice must choose odd x, leaving Bob with even n-x
    # - Even numbers always have the divisor 1, odd numbers >1 have no even divisors
    # - Player receiving even number can always force opponent to receive odd number
    # - Player receiving 1 loses (no valid moves)
    # Therefore: Alice wins iff n is even
    
    return n % 2 == 0


def divisor_game_recursive(n):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to explore all possible moves.
    
    Time Complexity: O(n!) - factorial explosion
    Space Complexity: O(n) - recursion stack
    """
    def can_win(current_n):
        """Return True if current player can win from state current_n"""
        if current_n == 1:
            return False  # No moves available, current player loses
        
        # Try all possible divisors
        for x in range(1, current_n):
            if current_n % x == 0:
                # If opponent can't win from (current_n - x), current player wins
                if not can_win(current_n - x):
                    return True
        
        return False  # No winning move found
    
    return can_win(n)


def divisor_game_memoization(n):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed states.
    
    Time Complexity: O(n^2) - each state computed once
    Space Complexity: O(n) - memoization table
    """
    memo = {}
    
    def can_win(current_n):
        if current_n == 1:
            return False
        
        if current_n in memo:
            return memo[current_n]
        
        # Try all possible divisors
        for x in range(1, current_n):
            if current_n % x == 0:
                if not can_win(current_n - x):
                    memo[current_n] = True
                    return True
        
        memo[current_n] = False
        return False
    
    return can_win(n)


def divisor_game_dp_bottom_up(n):
    """
    BOTTOM-UP DP APPROACH:
    =====================
    Build solutions from small to large numbers.
    
    Time Complexity: O(n^2) - nested loops
    Space Complexity: O(n) - DP array
    """
    if n == 1:
        return False
    
    # dp[i] = True if current player can win starting with number i
    dp = [False] * (n + 1)
    dp[1] = False  # Base case: no moves from 1
    
    for i in range(2, n + 1):
        # Try all possible divisors of i
        for x in range(1, i):
            if i % x == 0:
                # If opponent can't win from (i - x), current player wins
                if not dp[i - x]:
                    dp[i] = True
                    break
    
    return dp[n]


def divisor_game_with_analysis(n):
    """
    DIVISOR GAME WITH DETAILED ANALYSIS:
    ===================================
    Solve the game and provide comprehensive insights.
    
    Time Complexity: O(n^2) - DP computation + analysis
    Space Complexity: O(n) - DP table + analysis data
    """
    analysis = {
        'n': n,
        'mathematical_result': n % 2 == 0,
        'winning_strategy': None,
        'optimal_moves': [],
        'divisor_analysis': {},
        'game_tree_insights': [],
        'parity_patterns': {}
    }
    
    # Analyze divisors
    divisors = [x for x in range(1, n) if n % x == 0]
    analysis['divisor_analysis'] = {
        'divisors': divisors,
        'count': len(divisors),
        'even_divisors': [x for x in divisors if x % 2 == 0],
        'odd_divisors': [x for x in divisors if x % 2 == 1]
    }
    
    # DP with move tracking
    if n == 1:
        analysis['winning_strategy'] = "Alice loses immediately (no moves)"
        return False, analysis
    
    dp = [False] * (n + 1)
    best_moves = [0] * (n + 1)
    dp[1] = False
    
    for i in range(2, n + 1):
        winning_move = None
        
        for x in range(1, i):
            if i % x == 0:
                if not dp[i - x]:
                    dp[i] = True
                    winning_move = x
                    break
        
        best_moves[i] = winning_move if winning_move else 0
    
    # Reconstruct optimal strategy for Alice
    if dp[n]:
        strategy = []
        current = n
        is_alice_turn = True
        
        while current > 1 and best_moves[current] != 0:
            move = best_moves[current]
            strategy.append({
                'player': 'Alice' if is_alice_turn else 'Bob',
                'from': current,
                'subtract': move,
                'to': current - move,
                'reasoning': f"Choose divisor {move} of {current}"
            })
            
            current -= move
            is_alice_turn = not is_alice_turn
        
        analysis['optimal_moves'] = strategy
        analysis['winning_strategy'] = f"Alice wins by choosing {best_moves[n]} first"
    else:
        analysis['winning_strategy'] = "Alice has no winning strategy"
    
    # Parity analysis
    even_count = sum(1 for i in range(2, n + 1) if dp[i] and i % 2 == 0)
    odd_count = sum(1 for i in range(2, n + 1) if dp[i] and i % 2 == 1)
    
    analysis['parity_patterns'] = {
        'even_winning_positions': even_count,
        'odd_winning_positions': odd_count,
        'pattern_confirmation': even_count > 0 and odd_count == 0
    }
    
    # Game theory insights
    if n % 2 == 0:
        analysis['game_tree_insights'].append("Even n: Alice can always choose 1, forcing Bob to odd position")
        analysis['game_tree_insights'].append("Odd positions have only odd divisors, leading to even positions")
        analysis['game_tree_insights'].append("This creates a forced win for Alice")
    else:
        analysis['game_tree_insights'].append("Odd n: Alice must choose odd divisor, giving Bob even position")
        analysis['game_tree_insights'].append("Bob can then apply the even-position winning strategy")
        analysis['game_tree_insights'].append("Alice is forced into a losing position")
    
    return dp[n], analysis


def divisor_game_analysis(n):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the divisor game with detailed mathematical insights.
    """
    print(f"Divisor Game Analysis for n = {n}:")
    print(f"Starting number: {n}")
    print(f"Parity: {'Even' if n % 2 == 0 else 'Odd'}")
    
    # Find all divisors
    divisors = [x for x in range(1, n) if n % x == 0]
    print(f"Valid moves (divisors < n): {divisors}")
    print(f"Number of possible first moves: {len(divisors)}")
    
    # Different approaches
    mathematical = divisor_game_mathematical_insight(n)
    
    if n <= 20:  # Only for small n due to factorial complexity
        try:
            recursive = divisor_game_recursive(n)
            print(f"Recursive result: {recursive}")
        except:
            print("Recursive: Too slow")
    
    memoization = divisor_game_memoization(n)
    bottom_up = divisor_game_dp_bottom_up(n)
    
    print(f"Mathematical insight: {mathematical}")
    print(f"Memoization result: {memoization}")
    print(f"Bottom-up DP result: {bottom_up}")
    
    # Detailed analysis
    detailed_result, analysis = divisor_game_with_analysis(n)
    
    print(f"\nDetailed Analysis:")
    print(f"Alice can win: {detailed_result}")
    print(f"Winning strategy: {analysis['winning_strategy']}")
    
    print(f"\nDivisor Analysis:")
    div_analysis = analysis['divisor_analysis']
    print(f"All divisors: {div_analysis['divisors']}")
    print(f"Even divisors: {div_analysis['even_divisors']}")
    print(f"Odd divisors: {div_analysis['odd_divisors']}")
    
    if analysis['optimal_moves']:
        print(f"\nOptimal Game Sequence:")
        for i, move in enumerate(analysis['optimal_moves']):
            print(f"  Move {i+1}: {move['player']} at {move['from']} chooses {move['subtract']} → {move['to']}")
            print(f"           {move['reasoning']}")
    
    print(f"\nParity Pattern Analysis:")
    parity = analysis['parity_patterns']
    print(f"Even positions that are winning: {parity['even_winning_positions']}")
    print(f"Odd positions that are winning: {parity['odd_winning_positions']}")
    print(f"Pattern matches theory: {parity['pattern_confirmation']}")
    
    print(f"\nGame Theory Insights:")
    for insight in analysis['game_tree_insights']:
        print(f"  • {insight}")
    
    # Mathematical proof explanation
    print(f"\nMathematical Proof:")
    print(f"1. If n is even, Alice can choose x=1 (always a divisor)")
    print(f"2. This leaves Bob with n-1, which is odd")
    print(f"3. Odd numbers only have odd divisors")
    print(f"4. So Bob must choose odd x, leaving Alice with even number")
    print(f"5. Alice can repeat this strategy until Bob gets 1")
    print(f"6. Therefore: Even n → Alice wins, Odd n → Bob wins")
    
    return detailed_result


def divisor_game_variants():
    """
    DIVISOR GAME VARIANTS:
    =====================
    Different rule modifications and extensions.
    """
    
    def divisor_game_prime_only(n):
        """Divisor game where only prime divisors can be chosen"""
        def get_prime_divisors(num):
            primes = []
            for i in range(2, num):
                if num % i == 0:
                    # Check if i is prime
                    is_prime = True
                    for j in range(2, int(i**0.5) + 1):
                        if i % j == 0:
                            is_prime = False
                            break
                    if is_prime:
                        primes.append(i)
            return primes
        
        memo = {}
        
        def can_win(current_n):
            if current_n == 1:
                return False
            
            if current_n in memo:
                return memo[current_n]
            
            prime_divisors = get_prime_divisors(current_n)
            if not prime_divisors:
                memo[current_n] = False
                return False
            
            for p in prime_divisors:
                if not can_win(current_n - p):
                    memo[current_n] = True
                    return True
            
            memo[current_n] = False
            return False
        
        return can_win(n)
    
    def divisor_game_multiple_subtract(n, k):
        """Divisor game where you can subtract any multiple of chosen divisor"""
        memo = {}
        
        def can_win(current_n):
            if current_n == 1:
                return False
            
            if current_n in memo:
                return memo[current_n]
            
            for x in range(1, current_n):
                if current_n % x == 0:
                    # Can subtract any multiple of x
                    for multiple in range(1, k + 1):
                        if current_n - x * multiple > 0:
                            if not can_win(current_n - x * multiple):
                                memo[current_n] = True
                                return True
            
            memo[current_n] = False
            return False
        
        return can_win(n)
    
    def divisor_game_three_players(n):
        """Approximate three-player divisor game"""
        # Simplified analysis for three players
        # In multi-player games, the dynamics change significantly
        if n <= 3:
            return "Player 1 wins" if n == 2 else "Player 2/3 advantage"
        
        # With three players, even/odd analysis becomes more complex
        # This is a simplified heuristic
        if n % 3 == 1:
            return "Player 1 disadvantage"
        else:
            return "Player 1 has some advantage"
    
    def divisor_game_with_costs(n, cost_per_move):
        """Divisor game where each move has a cost"""
        # Modified game where moves have costs
        # Player wants to win while minimizing cost
        memo = {}
        
        def min_cost_to_win(current_n):
            if current_n == 1:
                return float('inf')  # Can't win from 1
            
            if current_n in memo:
                return memo[current_n]
            
            min_cost = float('inf')
            
            for x in range(1, current_n):
                if current_n % x == 0:
                    opponent_cost = min_cost_to_win(current_n - x)
                    if opponent_cost == float('inf'):
                        # Opponent can't win, so current player wins
                        min_cost = min(min_cost, cost_per_move)
                    # If opponent can win, current player might still try other moves
            
            memo[current_n] = min_cost
            return min_cost
        
        cost = min_cost_to_win(n)
        return cost < float('inf'), cost if cost < float('inf') else 0
    
    # Test variants
    test_values = [2, 3, 4, 5, 6, 10, 12, 15]
    
    print("Divisor Game Variants:")
    print("=" * 50)
    
    for n in test_values:
        print(f"\nn = {n}:")
        
        basic_result = divisor_game_mathematical_insight(n)
        print(f"Basic divisor game: Alice wins: {basic_result}")
        
        # Prime divisors only
        if n <= 12:  # Limit for complexity
            prime_result = divisor_game_prime_only(n)
            print(f"Prime divisors only: Alice wins: {prime_result}")
        
        # Multiple subtract
        if n <= 10:
            multiple_result = divisor_game_multiple_subtract(n, 2)
            print(f"Can subtract 2x divisor: Alice wins: {multiple_result}")
        
        # Three players
        three_player = divisor_game_three_players(n)
        print(f"Three players: {three_player}")
        
        # With costs
        if n <= 8:
            can_win_cost, cost = divisor_game_with_costs(n, 1)
            print(f"With move cost 1: Alice wins: {can_win_cost}, cost: {cost}")


# Test cases
def test_divisor_game():
    """Test all implementations with various inputs"""
    test_cases = [
        (1, False),
        (2, True),
        (3, False),
        (4, True),
        (5, False),
        (6, True),
        (7, False),
        (8, True),
        (9, False),
        (10, True)
    ]
    
    print("Testing Divisor Game Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}")
        print(f"Expected: {expected}")
        
        mathematical = divisor_game_mathematical_insight(n)
        
        # Skip recursive for larger inputs
        if n <= 15:
            try:
                recursive = divisor_game_recursive(n)
                print(f"Recursive:        {recursive} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = divisor_game_memoization(n)
        bottom_up = divisor_game_dp_bottom_up(n)
        
        print(f"Mathematical:     {mathematical} {'✓' if mathematical == expected else '✗'}")
        print(f"Memoization:      {memoization} {'✓' if memoization == expected else '✗'}")
        print(f"Bottom-up DP:     {bottom_up} {'✓' if bottom_up == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    divisor_game_analysis(6)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    divisor_game_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MATHEMATICAL PATTERN: Alice wins iff n is even")
    print("2. PARITY STRATEGY: Even positions can always force odd positions")
    print("3. OPTIMAL PLAY: Choose divisor 1 when n is even")
    print("4. GAME THEORY: Simple rule with deep mathematical foundation")
    print("5. PROOF TECHNIQUE: Invariant maintenance through parity")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Mathematical Games: Parity-based strategy analysis")
    print("• Algorithm Design: Pattern recognition in game states")
    print("• Competitive Programming: Quick mathematical insight problems")
    print("• Game Theory: Invariant-based winning strategy proofs")
    print("• Educational: Teaching proof techniques and pattern recognition")


if __name__ == "__main__":
    test_divisor_game()


"""
DIVISOR GAME - MATHEMATICAL GAME THEORY WITH PARITY ANALYSIS:
=============================================================

This problem demonstrates elegant mathematical game theory:
- Simple rules with profound mathematical pattern (parity-based strategy)
- Closed-form solution derived from game theory principles
- Educational example of invariant-based winning strategy proofs
- Contrast between computational and mathematical approaches

KEY INSIGHTS:
============
1. **MATHEMATICAL PATTERN**: Alice wins if and only if n is even
2. **PARITY STRATEGY**: Player with even number can always force opponent to odd number
3. **INVARIANT MAINTENANCE**: Even → odd → even cycle guarantees Alice's victory
4. **OPTIMAL PLAY**: When n is even, always choose divisor 1 (forcing odd position)
5. **PROOF ELEGANCE**: Simple mathematical insight eliminates need for complex computation

ALGORITHM APPROACHES:
====================

1. **Mathematical Insight**: O(1) time, O(1) space
   - Recognize that Alice wins iff n is even
   - Most elegant solution based on mathematical proof

2. **Recursive**: O(n!) time, O(n) space
   - Brute force exploration of all game trees
   - Exponential complexity without memoization

3. **Memoization**: O(n²) time, O(n) space
   - Top-down DP with state caching
   - Practical for moderate n values

4. **Bottom-up DP**: O(n²) time, O(n) space
   - Iterative DP building winning states
   - Educational for understanding game progression

MATHEMATICAL PROOF:
==================
**Theorem**: Alice wins the divisor game if and only if n is even.

**Proof**:
```
Case 1: n is even
- Alice can choose x = 1 (since n % 1 = 0 always)
- This leaves Bob with n - 1, which is odd
- Bob must choose some divisor y of (n-1)
- Since n-1 is odd, all its divisors are odd
- So Bob leaves Alice with (n-1) - y = even - odd = odd
- Wait, this is wrong. Let me correct:

Correct Proof:
- If n is even, Alice chooses x = 1, leaving Bob with n-1 (odd)
- If n is odd, all proper divisors of n are odd (since odd = odd × odd)
- So Alice must choose odd x, leaving Bob with n-x = odd-odd = even
- Player with even number can always choose 1, forcing opponent to odd
- Player with odd number must choose odd divisor, giving opponent even
- Game ends when someone reaches 1 (losing position)
- Since Alice starts: Even n → Alice wins, Odd n → Bob wins
```

CORE MATHEMATICAL ALGORITHM:
============================
```python
def divisorGame(n):
    # Mathematical insight: Alice wins iff n is even
    return n % 2 == 0

# Proof verification through DP:
def divisorGameDP(n):
    if n == 1:
        return False
    
    dp = [False] * (n + 1)
    dp[1] = False  # Base case
    
    for i in range(2, n + 1):
        for x in range(1, i):
            if i % x == 0:  # x is divisor of i
                if not dp[i - x]:  # Opponent loses from i-x
                    dp[i] = True
                    break
    
    return dp[n]
```

PARITY ANALYSIS:
===============
**Even Numbers**: 
- Always have divisor 1
- Can force opponent to odd position
- Maintain winning advantage

**Odd Numbers**:
- All proper divisors are odd  
- Must give opponent even position
- Cannot escape losing cycle

**Strategy**: Player with even number always chooses x=1

GAME TREE STRUCTURE:
===================
**Branching Factor**: Variable (number of divisors)
**Tree Depth**: Variable (depends on chosen moves)
**Winning Pattern**: Alternating even/odd positions determine outcome
**Optimal Strategy**: Simple greedy choice (x=1) when possible

INVARIANT MAINTENANCE:
=====================
**Invariant**: "Player receiving even number has winning strategy"

**Maintenance**:
- Even position: Choose x=1 → opponent gets odd position
- Odd position: Must choose odd x → opponent gets even position  
- Invariant preserved through optimal play

COMPLEXITY ANALYSIS:
===================
**Mathematical Solution**: O(1) time, O(1) space
**DP Solution**: O(n²) time for educational purposes
**Space Optimization**: Can optimize DP to O(n) by avoiding repeated divisor computation

EDUCATIONAL VALUE:
=================
**Proof Techniques**:
- Invariant-based reasoning
- Parity arguments in game theory
- Mathematical insight vs computational approach

**Game Theory Concepts**:
- Winning/losing positions
- Strategy stealing
- Optimal play assumptions

APPLICATIONS:
============
- **Mathematical Competitions**: Pattern recognition in number theory games
- **Game Theory Education**: Teaching invariant-based proofs
- **Algorithm Design**: Recognizing when mathematical insight trumps computation
- **Competitive Programming**: Quick solution through mathematical analysis
- **Proof Techniques**: Demonstrating parity arguments

RELATED PROBLEMS:
================
- **Nim Game**: Another mathematical game with elegant solution
- **Stone Games**: Various resource allocation games
- **Number Theory Games**: Games based on mathematical properties
- **Invariant Games**: Games where maintaining invariants determines winner

VARIANTS:
========
- **Prime Divisors Only**: Restrict to prime divisor choices
- **Multiple Subtraction**: Allow subtracting multiples of divisors
- **Multi-Player**: Extension to more than two players
- **Cost-Based**: Add costs to moves for optimization objectives

EDGE CASES:
==========
- **n = 1**: No moves available, Alice loses immediately
- **n = 2**: Alice chooses 1, Bob gets 1 and loses
- **Small Primes**: Interesting cases for variant analysis
- **Large Even/Odd**: Pattern holds regardless of magnitude

This problem beautifully demonstrates how mathematical insight
can provide elegant solutions to seemingly complex game theory
problems, showing the power of pattern recognition and proof
techniques in competitive algorithm design.
"""
