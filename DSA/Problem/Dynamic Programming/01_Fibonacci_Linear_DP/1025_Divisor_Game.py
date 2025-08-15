"""
LeetCode 1025: Divisor Game
Difficulty: Easy
Category: Fibonacci & Linear DP / Game Theory

PROBLEM DESCRIPTION:
===================
Alice and Bob take turns playing a game, with Alice starting first.

Initially, there is a number n on the chalkboard. On each player's turn, that player makes a move consisting of:
- Choosing any x with 0 < x < n and n % x == 0.
- Replacing the number n on the chalkboard with n - x.

Also, if a player cannot make a move, they lose the game.

Return true if and only if Alice wins the game, assuming both players play optimally.

Example 1:
Input: n = 2
Output: true
Explanation: Alice chooses 1, Bob is left with 1 and loses.

Example 2:
Input: n = 3
Output: false
Explanation: Alice chooses 1, Bob chooses 1, and Alice has 1 left and loses.
Or Alice chooses 1, Bob chooses 1, and Alice has 1 left and loses.

Constraints:
- 1 <= n <= 1000
"""

def divisor_game_bruteforce(n):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible moves recursively using game theory.
    
    Time Complexity: O(n!) - extremely high due to game tree exploration
    Space Complexity: O(n) - recursion stack depth
    """
    def can_win(current_n):
        # Base case: if current player faces 1, they lose (no valid moves)
        if current_n == 1:
            return False
        
        # Try all possible moves (divisors of current_n)
        for x in range(1, current_n):
            if current_n % x == 0:
                # Make move: current_n becomes current_n - x
                # If opponent loses from that position, current player wins
                if not can_win(current_n - x):
                    return True
        
        # No winning move found
        return False
    
    return can_win(n)


def divisor_game_memoization(n):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to cache game state results.
    
    Time Complexity: O(n^2) - n states, each tries up to n moves
    Space Complexity: O(n) - memoization table + recursion stack
    """
    memo = {}
    
    def can_win(current_n):
        if current_n == 1:
            return False
        
        if current_n in memo:
            return memo[current_n]
        
        # Try all possible moves
        for x in range(1, current_n):
            if current_n % x == 0:
                # If opponent loses from next state, current player wins
                if not can_win(current_n - x):
                    memo[current_n] = True
                    return True
        
        memo[current_n] = False
        return False
    
    return can_win(n)


def divisor_game_tabulation(n):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using DP array.
    dp[i] = True if current player can win starting with number i
    
    Time Complexity: O(n^2) - nested loops for all numbers and divisors
    Space Complexity: O(n) - DP array
    """
    if n == 1:
        return False
    
    # dp[i] = True if current player wins starting with number i
    dp = [False] * (n + 1)
    dp[1] = False  # Base case: starting with 1, current player loses
    
    # Fill DP table from 2 to n
    for i in range(2, n + 1):
        # Try all possible moves (all divisors x where 0 < x < i)
        for x in range(1, i):
            if i % x == 0:
                # If opponent loses from state (i - x), current player wins
                if not dp[i - x]:
                    dp[i] = True
                    break  # Found a winning move, no need to check more
    
    return dp[n]


def divisor_game_mathematical_pattern(n):
    """
    MATHEMATICAL PATTERN RECOGNITION (OPTIMAL):
    ==========================================
    Discover the pattern: Alice wins if and only if n is even.
    
    Time Complexity: O(1) - constant time
    Space Complexity: O(1) - constant space
    """
    # Mathematical insight:
    # If n is even, Alice can choose x = 1 (since 1 divides all numbers)
    # This leaves Bob with odd n - 1
    # If n is odd, all proper divisors of n are odd
    # So Alice leaves Bob with even number (odd - odd = even)
    # The player with an even number can always force opponent to have odd
    # Eventually someone gets n = 1 (odd) and loses
    
    return n % 2 == 0


def divisor_game_with_proof():
    """
    MATHEMATICAL PROOF:
    ==================
    Prove why Alice wins if and only if n is even.
    """
    def explain_strategy():
        print("Mathematical Proof:")
        print("===================")
        print("Claim: Alice wins if and only if n is even.")
        print()
        print("Proof:")
        print("1. If n is even:")
        print("   - Alice can choose x = 1 (valid since 1 divides any number)")
        print("   - This leaves Bob with n - 1, which is odd")
        print("   - Now Bob has an odd number")
        print()
        print("2. If n is odd:")
        print("   - All proper divisors of odd numbers are odd")
        print("   - Alice must choose some odd x")
        print("   - This leaves Bob with n - x = odd - odd = even")
        print("   - Now Bob has an even number")
        print()
        print("3. Key insight:")
        print("   - Player with even number can always choose x = 1")
        print("   - This forces opponent to have odd number")
        print("   - Player with odd number must give opponent even number")
        print()
        print("4. Game progression:")
        print("   - If Alice starts with even n, she maintains control")
        print("   - Bob always gets odd numbers, Alice always gets even")
        print("   - Eventually Bob gets n = 1 and loses")
        print()
        print("Therefore: Alice wins ⟺ n is even")
    
    explain_strategy()
    return divisor_game_mathematical_pattern


def divisor_game_game_simulation(n):
    """
    GAME SIMULATION:
    ===============
    Simulate actual optimal gameplay and show moves.
    
    Time Complexity: O(n) - game length is at most n
    Space Complexity: O(1) - constant space for simulation
    """
    def get_optimal_move(current_n):
        """Find optimal move for current player"""
        if current_n % 2 == 0:
            # If even, choose x = 1 to give opponent odd number
            return 1
        else:
            # If odd, find any valid divisor (will give opponent even number)
            for x in range(1, current_n):
                if current_n % x == 0:
                    return x
            return None  # Should never happen for n > 1
    
    current_number = n
    alice_turn = True
    moves = []
    
    print(f"Game Simulation for n = {n}")
    print("=" * 40)
    print(f"Starting number: {n}")
    print()
    
    while current_number > 1:
        player = "Alice" if alice_turn else "Bob"
        move = get_optimal_move(current_number)
        
        if move is None:
            print(f"{player} cannot move and loses!")
            return not alice_turn
        
        new_number = current_number - move
        moves.append((player, current_number, move, new_number))
        print(f"{player}: n={current_number}, chooses x={move} (divisor), new n={new_number}")
        
        current_number = new_number
        alice_turn = not alice_turn
    
    # Whoever's turn it is now faces n=1 and loses
    loser = "Alice" if alice_turn else "Bob"
    winner = "Bob" if alice_turn else "Alice"
    
    print(f"\n{loser} faces n=1 and cannot move. {winner} wins!")
    print(f"Final result: Alice {'WINS' if winner == 'Alice' else 'LOSES'}")
    
    return winner == "Alice"


def divisor_game_complete_analysis(n):
    """
    COMPLETE GAME TREE ANALYSIS:
    ============================
    Analyze all possible moves for small n values.
    
    Time Complexity: O(2^n) - exponential game tree for analysis
    Space Complexity: O(n) - recursion depth
    """
    def analyze_position(current_n, depth=0, path=""):
        indent = "  " * depth
        
        if current_n == 1:
            print(f"{indent}n=1: Current player LOSES (no moves)")
            return False
        
        print(f"{indent}n={current_n}: Analyzing moves...")
        
        # Find all valid moves (divisors)
        valid_moves = []
        for x in range(1, current_n):
            if current_n % x == 0:
                valid_moves.append(x)
        
        print(f"{indent}Valid moves (divisors): {valid_moves}")
        
        winning_moves = []
        losing_moves = []
        
        for move in valid_moves:
            next_n = current_n - move
            print(f"{indent}  Try x={move} -> n={next_n}")
            
            # Opponent's result from next position
            opponent_wins = analyze_position(next_n, depth + 2, path + f" -> {move}")
            
            if not opponent_wins:  # If opponent loses, current player wins
                winning_moves.append(move)
                print(f"{indent}  x={move} is WINNING move")
            else:
                losing_moves.append(move)
                print(f"{indent}  x={move} is losing move")
        
        if winning_moves:
            print(f"{indent}Result: Current player WINS with moves {winning_moves}")
            return True
        else:
            print(f"{indent}Result: Current player LOSES (all moves: {losing_moves})")
            return False
    
    print(f"Complete Game Analysis for n = {n}")
    print("=" * 50)
    result = analyze_position(n)
    print(f"\nFinal Conclusion: Alice {'WINS' if result else 'LOSES'}")
    return result


def divisor_game_pattern_verification(max_n):
    """
    PATTERN VERIFICATION:
    ====================
    Verify the even/odd pattern for multiple values using DP.
    
    Time Complexity: O(max_n^3) - for each n, O(n^2) DP computation
    Space Complexity: O(max_n) - DP array
    """
    print("Pattern Verification: Alice wins ⟺ n is even")
    print("=" * 50)
    print("n\tDP Result\tEven?\tPattern Match\tExplanation")
    print("-" * 70)
    
    all_correct = True
    
    for n in range(1, max_n + 1):
        # Calculate using DP
        dp_result = divisor_game_tabulation(n)
        
        # Check mathematical pattern
        is_even = (n % 2 == 0)
        pattern_match = (dp_result == is_even)
        
        explanation = ""
        if n == 1:
            explanation = "Base case: cannot move"
        elif is_even:
            explanation = "Even: Alice chooses 1, Bob gets odd"
        else:
            explanation = "Odd: Alice gives Bob even number"
        
        status = "✓" if pattern_match else "✗"
        print(f"{n}\t{dp_result}\t\t{is_even}\t{status}\t\t{explanation}")
        
        if not pattern_match:
            all_correct = False
    
    print(f"\nPattern Verification: {'PASSED' if all_correct else 'FAILED'}")
    print(f"Mathematical pattern holds for all tested values: {all_correct}")
    
    return all_correct


# Test cases
def test_divisor_game():
    """Test all implementations with various inputs"""
    test_cases = [
        (1, False),   # Base case
        (2, True),    # Alice: 2 -> 1, Bob loses
        (3, False),   # Alice: 3 -> 2, Bob: 2 -> 1, Alice loses
        (4, True),    # Alice: 4 -> 3, Bob: 3 -> 2, Alice: 2 -> 1, Bob loses
        (5, False),   # Odd number, Alice loses
        (6, True),    # Even number, Alice wins
        (7, False),   # Odd number, Alice loses
        (8, True),    # Even number, Alice wins
        (9, False),   # Odd number, Alice loses
        (10, True),   # Even number, Alice wins
        (100, True),  # Large even number
        (101, False)  # Large odd number
    ]
    
    print("Testing Divisor Game Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n = {n}")
        print(f"Expected: {expected} (Alice {'wins' if expected else 'loses'})")
        
        # Test all approaches
        if n <= 10:  # Skip brute force for larger n due to exponential complexity
            brute = divisor_game_bruteforce(n)
            print(f"Brute Force:      {brute} {'✓' if brute == expected else '✗'}")
        
        memo = divisor_game_memoization(n)
        tab = divisor_game_tabulation(n)
        pattern = divisor_game_mathematical_pattern(n)
        
        print(f"Memoization:      {memo} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab} {'✓' if tab == expected else '✗'}")
        print(f"Math Pattern:     {pattern} {'✓' if pattern == expected else '✗'}")
    
    # Detailed analysis for small numbers
    print(f"\n" + "=" * 70)
    print("DETAILED GAME SIMULATIONS:")
    
    for n in [2, 3, 4, 5]:
        print(f"\n{'-' * 50}")
        divisor_game_game_simulation(n)
    
    # Pattern verification
    print(f"\n" + "=" * 70)
    print("PATTERN VERIFICATION:")
    divisor_game_pattern_verification(12)
    
    # Complete analysis for very small numbers
    print(f"\n" + "=" * 70)
    print("COMPLETE GAME TREE ANALYSIS:")
    
    for n in [2, 3]:
        print(f"\n{'-' * 50}")
        divisor_game_complete_analysis(n)
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(n!),       Space: O(n)")
    print("Memoization:      Time: O(n^2),      Space: O(n)")
    print("Tabulation:       Time: O(n^2),      Space: O(n)")
    print("Math Pattern:     Time: O(1),        Space: O(1)")
    print("Game Simulation:  Time: O(n),        Space: O(1)")


if __name__ == "__main__":
    test_divisor_game()


"""
PATTERN RECOGNITION:
==================
This is a classic game theory DP problem:
- Two players with optimal play
- Game has perfect information
- Determine if first player (Alice) can win
- Mathematical pattern emerges: Alice wins ⟺ n is even

KEY INSIGHT - MATHEMATICAL DISCOVERY:
====================================
Through DP analysis, we discover a simple pattern:
Alice wins if and only if n is even.

Mathematical Proof:
1. If n is even: Alice chooses x=1, leaving Bob with odd n-1
2. If n is odd: Any divisor x is odd, leaving Bob with even n-x
3. Player with even number can always force opponent to have odd
4. Game ends when someone gets n=1 (odd) and loses
5. Since Alice starts, she wins ⟺ she can maintain even numbers

GAME THEORY CONCEPTS:
====================
1. **Perfect Information**: Both players see complete game state
2. **Zero-Sum Game**: One player wins, other loses
3. **Optimal Play**: Both players play perfectly
4. **Backward Induction**: Analyze game tree from end states
5. **Winning/Losing Positions**: Classify game states

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(n!) - explore entire game tree
2. **Memoization**: O(n²) - cache repeated game states  
3. **Tabulation**: O(n²) - bottom-up DP computation
4. **Pattern Recognition**: O(1) - mathematical insight

The mathematical pattern makes DP unnecessary, but DP helps discover the pattern!

STATE DEFINITION:
================
dp[i] = True if current player can win starting with number i

RECURRENCE RELATION:
===================
dp[i] = True if there exists divisor x of i such that dp[i-x] = False
Base case: dp[1] = False (no valid moves)

MATHEMATICAL PROPERTIES:
=======================
1. **Odd numbers**: All proper divisors are odd
2. **Even numbers**: Has divisor 1 (and possibly even divisors)
3. **Parity preservation**: odd - odd = even, even - odd = odd
4. **Control strategy**: Player with even n can force opponent to odd

VARIANTS TO PRACTICE:
====================
- Stone Game (877) - similar optimal play analysis
- Nim Game (292) - another pattern-based game
- Predict the Winner (486) - array-based game theory
- Stone Game II (1140) - more complex stone removal

INTERVIEW TIPS:
==============
1. Start with game theory DP approach
2. Implement memoization to handle overlapping states
3. Analyze small cases to find patterns
4. Prove mathematical pattern rigorously
5. Optimize to O(1) constant-time solution
6. Explain optimal strategies for both players
7. Discuss why pattern works (parity arguments)
8. Show game simulation for concrete examples
"""
