"""
LeetCode 464: Can I Win
Difficulty: Medium
Category: Game Theory DP - State-Based Minimax

PROBLEM DESCRIPTION:
===================
In the "100 game" two players take turns adding, to a running total, any integer from 1 to maxChoosableInteger. The player who first causes the running total to reach or exceed desiredTotal wins.

What is the best strategy for the first player? You are to determine whether the first player can force a win, assuming both players play optimally.

For example, if maxChoosableInteger = 10 and desiredTotal = 11, then the first player can win by choosing 1 first. Regardless of what the second player chooses, the first player can choose a number that causes the total to reach or exceed 11.

Example 1:
Input: maxChoosableInteger = 10, desiredTotal = 11
Output: true
Explanation: The first player can always win by choosing 1.

Example 2:
Input: maxChoosableInteger = 10, desiredTotal = 40
Output: false
Explanation: Whether the first player chooses 1 or 10, the second player can always make the first player lose.

Constraints:
- 1 <= maxChoosableInteger <= 20
- 1 <= desiredTotal <= 300
"""


def can_i_win_recursive(maxChoosableInteger, desiredTotal):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to explore all possible game states.
    
    Time Complexity: O(2^n * m) where n=maxChoosableInteger, m=desiredTotal
    Space Complexity: O(n) - recursion stack
    """
    # Quick checks
    if maxChoosableInteger >= desiredTotal:
        return True
    
    if sum(range(1, maxChoosableInteger + 1)) < desiredTotal:
        return False
    
    def can_win(used_numbers, current_total):
        """
        Check if current player can win from this state
        used_numbers: set of already used numbers
        current_total: current running total
        """
        # Try each available number
        for num in range(1, maxChoosableInteger + 1):
            if num in used_numbers:
                continue
            
            # If this number wins immediately
            if current_total + num >= desiredTotal:
                return True
            
            # If opponent cannot win after this move
            used_numbers.add(num)
            if not can_win(used_numbers, current_total + num):
                used_numbers.remove(num)
                return True
            used_numbers.remove(num)
        
        return False
    
    return can_win(set(), 0)


def can_i_win_memoization(maxChoosableInteger, desiredTotal):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed states.
    
    Time Complexity: O(2^n * m) - with memoization
    Space Complexity: O(2^n) - memoization table
    """
    # Early termination checks
    if maxChoosableInteger >= desiredTotal:
        return True
    
    total_sum = maxChoosableInteger * (maxChoosableInteger + 1) // 2
    if total_sum < desiredTotal:
        return False
    
    memo = {}
    
    def can_win(used_mask, current_total):
        """
        Check if current player can win
        used_mask: bitmask representing used numbers
        current_total: current running total
        """
        if used_mask in memo:
            return memo[used_mask]
        
        # Try each available number
        for num in range(1, maxChoosableInteger + 1):
            if used_mask & (1 << num):
                continue
            
            # If this number wins immediately
            if current_total + num >= desiredTotal:
                memo[used_mask] = True
                return True
            
            # If opponent cannot win after this move
            new_mask = used_mask | (1 << num)
            if not can_win(new_mask, current_total + num):
                memo[used_mask] = True
                return True
        
        memo[used_mask] = False
        return False
    
    return can_win(0, 0)


def can_i_win_optimized_memoization(maxChoosableInteger, desiredTotal):
    """
    OPTIMIZED MEMOIZATION:
    =====================
    Optimize by using only bitmask (current total can be computed).
    
    Time Complexity: O(2^n) - each state computed once
    Space Complexity: O(2^n) - memoization table
    """
    # Quick feasibility checks
    if maxChoosableInteger >= desiredTotal:
        return True
    
    total_sum = maxChoosableInteger * (maxChoosableInteger + 1) // 2
    if total_sum < desiredTotal:
        return False
    
    memo = {}
    
    def can_win(used_mask):
        """
        Check if current player can win from this state
        used_mask: bitmask of used numbers
        """
        if used_mask in memo:
            return memo[used_mask]
        
        # Calculate current total from bitmask
        current_total = sum(i for i in range(1, maxChoosableInteger + 1) 
                          if used_mask & (1 << i))
        
        # Try each available number
        for num in range(1, maxChoosableInteger + 1):
            if used_mask & (1 << num):
                continue
            
            # If this number wins immediately
            if current_total + num >= desiredTotal:
                memo[used_mask] = True
                return True
            
            # If opponent cannot win after this move
            new_mask = used_mask | (1 << num)
            if not can_win(new_mask):
                memo[used_mask] = True
                return True
        
        memo[used_mask] = False
        return False
    
    return can_win(0)


def can_i_win_with_analysis(maxChoosableInteger, desiredTotal):
    """
    CAN I WIN WITH DETAILED ANALYSIS:
    ================================
    Solve the game and provide detailed strategic insights.
    
    Time Complexity: O(2^n) - memoized minimax
    Space Complexity: O(2^n) - memoization + analysis
    """
    analysis = {
        'max_choosable': maxChoosableInteger,
        'desired_total': desiredTotal,
        'total_possible_sum': maxChoosableInteger * (maxChoosableInteger + 1) // 2,
        'state_space_size': 2 ** maxChoosableInteger,
        'winning_strategy': None,
        'game_length_bounds': None,
        'critical_numbers': [],
        'strategy_insights': []
    }
    
    # Basic feasibility analysis
    if maxChoosableInteger >= desiredTotal:
        analysis['winning_strategy'] = f"Immediate win by choosing {desiredTotal}"
        analysis['strategy_insights'].append("First player can win in one move")
        return True, analysis
    
    if analysis['total_possible_sum'] < desiredTotal:
        analysis['winning_strategy'] = "Impossible - insufficient total sum"
        analysis['strategy_insights'].append("Not enough numbers to reach desired total")
        return False, analysis
    
    # Analyze game length bounds
    min_moves = (desiredTotal + maxChoosableInteger - 1) // maxChoosableInteger  # ceiling division
    max_moves = desiredTotal  # if we only choose 1 each time
    analysis['game_length_bounds'] = (min_moves, min(max_moves, maxChoosableInteger))
    
    # Find critical numbers (those that can win immediately from empty state)
    for num in range(max(1, desiredTotal - maxChoosableInteger + 1), maxChoosableInteger + 1):
        if num >= desiredTotal:
            analysis['critical_numbers'].append(num)
    
    # Memoized game solving with strategy tracking
    memo = {}
    best_moves = {}
    
    def can_win(used_mask):
        if used_mask in memo:
            return memo[used_mask]
        
        current_total = sum(i for i in range(1, maxChoosableInteger + 1) 
                          if used_mask & (1 << i))
        
        winning_moves = []
        
        for num in range(1, maxChoosableInteger + 1):
            if used_mask & (1 << num):
                continue
            
            # Immediate win
            if current_total + num >= desiredTotal:
                winning_moves.append(num)
                memo[used_mask] = True
                best_moves[used_mask] = num
                return True
            
            # Strategic win (opponent loses)
            new_mask = used_mask | (1 << num)
            if not can_win(new_mask):
                winning_moves.append(num)
        
        if winning_moves:
            memo[used_mask] = True
            best_moves[used_mask] = min(winning_moves)  # Choose smallest winning move
            return True
        else:
            memo[used_mask] = False
            return False
    
    result = can_win(0)
    
    # Strategy reconstruction
    if result:
        analysis['winning_strategy'] = f"First move: {best_moves.get(0, 'Unknown')}"
        
        # Analyze first move options
        first_move_options = []
        for num in range(1, maxChoosableInteger + 1):
            if num >= desiredTotal:
                first_move_options.append((num, "Immediate win"))
            else:
                new_mask = 1 << num
                if new_mask in memo and not memo[new_mask]:
                    first_move_options.append((num, "Strategic win"))
        
        if first_move_options:
            analysis['strategy_insights'].append(f"Winning first moves: {first_move_options}")
    else:
        analysis['winning_strategy'] = "No winning strategy exists"
        analysis['strategy_insights'].append("Second player has optimal counter-strategy")
    
    # Game complexity analysis
    states_explored = len(memo)
    analysis['states_explored'] = states_explored
    analysis['state_space_efficiency'] = states_explored / analysis['state_space_size']
    
    return result, analysis


def can_i_win_analysis(maxChoosableInteger, desiredTotal):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the game with detailed strategic insights.
    """
    print(f"Can I Win Game Analysis:")
    print(f"Max choosable integer: {maxChoosableInteger}")
    print(f"Desired total: {desiredTotal}")
    print(f"Available numbers: {list(range(1, maxChoosableInteger + 1))}")
    
    total_sum = maxChoosableInteger * (maxChoosableInteger + 1) // 2
    print(f"Sum of all numbers: {total_sum}")
    print(f"State space size: 2^{maxChoosableInteger} = {2**maxChoosableInteger:,}")
    
    # Basic feasibility
    if maxChoosableInteger >= desiredTotal:
        print(f"Trivial win: Can choose {desiredTotal} on first move")
        return True
    
    if total_sum < desiredTotal:
        print(f"Impossible: Total sum {total_sum} < desired {desiredTotal}")
        return False
    
    # Different approaches
    recursive = can_i_win_recursive(maxChoosableInteger, desiredTotal)
    memoization = can_i_win_memoization(maxChoosableInteger, desiredTotal)
    optimized = can_i_win_optimized_memoization(maxChoosableInteger, desiredTotal)
    
    print(f"Recursive result: {recursive}")
    print(f"Memoization result: {memoization}")
    print(f"Optimized result: {optimized}")
    
    # Detailed analysis
    detailed_result, analysis = can_i_win_with_analysis(maxChoosableInteger, desiredTotal)
    
    print(f"\nDetailed Analysis:")
    print(f"First player can win: {detailed_result}")
    print(f"Winning strategy: {analysis['winning_strategy']}")
    
    if analysis['game_length_bounds']:
        min_moves, max_moves = analysis['game_length_bounds']
        print(f"Game length bounds: {min_moves} to {max_moves} moves")
    
    if analysis['critical_numbers']:
        print(f"Critical numbers (immediate win): {analysis['critical_numbers']}")
    
    print(f"States explored: {analysis['states_explored']:,} / {analysis['state_space_size']:,}")
    print(f"State space efficiency: {analysis['state_space_efficiency']:.2%}")
    
    print(f"\nStrategy Insights:")
    for insight in analysis['strategy_insights']:
        print(f"  • {insight}")
    
    # Game theory properties
    print(f"\nGame Properties:")
    print(f"  • Perfect information: Both players know all available numbers")
    print(f"  • Zero-sum: One player's win is the other's loss")
    print(f"  • Finite: Game must end within {maxChoosableInteger} moves")
    print(f"  • Deterministic: No randomness involved")
    
    return detailed_result


def can_i_win_variants():
    """
    CAN I WIN VARIANTS:
    ==================
    Different game scenarios and modifications.
    """
    
    def can_i_win_with_negative_numbers(maxChoosableInteger, desiredTotal, hasNegatives=False):
        """Version where negative numbers are also choosable"""
        if not hasNegatives:
            return can_i_win_optimized_memoization(maxChoosableInteger, desiredTotal)
        
        # With negative numbers, the game becomes much more complex
        # Simplified version - would need more sophisticated analysis
        if maxChoosableInteger >= desiredTotal:
            return True
        
        # If we can choose negative numbers, the game dynamics change significantly
        return True  # Simplified - first player usually has advantage
    
    def can_i_win_multiplayer(maxChoosableInteger, desiredTotal, num_players):
        """Multi-player version (simplified)"""
        if num_players == 2:
            return can_i_win_optimized_memoization(maxChoosableInteger, desiredTotal)
        
        # For more players, use approximation
        # First player advantage decreases with more players
        total_sum = maxChoosableInteger * (maxChoosableInteger + 1) // 2
        
        if maxChoosableInteger >= desiredTotal:
            return True  # Can win immediately
        
        if total_sum < desiredTotal:
            return False  # Impossible
        
        # Rough heuristic: first player advantage decreases with more players
        return desiredTotal <= total_sum // num_players + maxChoosableInteger
    
    def can_i_win_with_repeated_use(maxChoosableInteger, desiredTotal):
        """Version where numbers can be reused"""
        # With reuse, first player can always win if maxChoosableInteger >= desiredTotal
        if maxChoosableInteger >= desiredTotal:
            return True
        
        # Otherwise, complex analysis needed
        # Simplified: if we can reach any multiple of maxChoosableInteger
        return desiredTotal % maxChoosableInteger != 0
    
    def can_i_win_range_restriction(maxChoosableInteger, desiredTotal, forbidden_numbers):
        """Version with some numbers forbidden"""
        available_numbers = [i for i in range(1, maxChoosableInteger + 1) 
                           if i not in forbidden_numbers]
        
        if not available_numbers:
            return False
        
        max_available = max(available_numbers)
        if max_available >= desiredTotal:
            return True
        
        total_available = sum(available_numbers)
        if total_available < desiredTotal:
            return False
        
        # Use modified memoization with available numbers only
        memo = {}
        
        def can_win(used_mask):
            if used_mask in memo:
                return memo[used_mask]
            
            current_total = sum(num for num in available_numbers 
                              if used_mask & (1 << num))
            
            for num in available_numbers:
                if used_mask & (1 << num):
                    continue
                
                if current_total + num >= desiredTotal:
                    memo[used_mask] = True
                    return True
                
                new_mask = used_mask | (1 << num)
                if not can_win(new_mask):
                    memo[used_mask] = True
                    return True
            
            memo[used_mask] = False
            return False
        
        return can_win(0)
    
    # Test variants
    test_cases = [
        (10, 11),
        (10, 40),
        (4, 6),
        (20, 50)
    ]
    
    print("Can I Win Variants:")
    print("=" * 50)
    
    for maxInt, desiredTotal in test_cases:
        print(f"\nMax Integer: {maxInt}, Desired Total: {desiredTotal}")
        
        basic_result = can_i_win_optimized_memoization(maxInt, desiredTotal)
        print(f"Basic game: First player wins: {basic_result}")
        
        # Negative numbers variant
        negative_result = can_i_win_with_negative_numbers(maxInt, desiredTotal, True)
        print(f"With negative numbers: First player wins: {negative_result}")
        
        # Multi-player variants
        for players in [3, 4]:
            multi_result = can_i_win_multiplayer(maxInt, desiredTotal, players)
            print(f"With {players} players: First player wins: {multi_result}")
        
        # Repeated use variant
        repeated_result = can_i_win_with_repeated_use(maxInt, desiredTotal)
        print(f"With number reuse: First player wins: {repeated_result}")
        
        # Forbidden numbers variant
        forbidden = [1, 2] if maxInt > 2 else []
        if forbidden:
            restricted_result = can_i_win_range_restriction(maxInt, desiredTotal, forbidden)
            print(f"Forbidding {forbidden}: First player wins: {restricted_result}")


# Test cases
def test_can_i_win():
    """Test all implementations with various inputs"""
    test_cases = [
        (10, 11, True),
        (10, 40, False),
        (4, 6, True),
        (1, 1, True),
        (2, 3, False),
        (3, 4, True),
        (20, 210, False),
        (20, 100, True)
    ]
    
    print("Testing Can I Win Solutions:")
    print("=" * 70)
    
    for i, (maxInt, desiredTotal, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"maxChoosableInteger = {maxInt}, desiredTotal = {desiredTotal}")
        print(f"Expected: {expected}")
        
        # Skip recursive for large cases
        if maxInt <= 15:
            try:
                recursive = can_i_win_recursive(maxInt, desiredTotal)
                print(f"Recursive:        {recursive} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = can_i_win_memoization(maxInt, desiredTotal)
        optimized = can_i_win_optimized_memoization(maxInt, desiredTotal)
        
        print(f"Memoization:      {memoization} {'✓' if memoization == expected else '✗'}")
        print(f"Optimized:        {optimized} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    can_i_win_analysis(10, 40)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    can_i_win_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. BITMASK STATE: Represent used numbers efficiently with bitmasks")
    print("2. MEMOIZATION: Cache game states to avoid recomputation")
    print("3. OPTIMAL PLAY: Both players play to maximize their win probability")
    print("4. STATE SPACE: Exponential in number of choosable integers")
    print("5. EARLY TERMINATION: Immediate wins and impossibility detection")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Game AI: Strategy computation for turn-based number games")
    print("• Decision Theory: Optimal choice under competitive scenarios")
    print("• Combinatorial Game Theory: Analysis of finite combinatorial games")
    print("• Resource Allocation: Competitive resource selection strategies")
    print("• Algorithm Design: State-based minimax optimization")


if __name__ == "__main__":
    test_can_i_win()


"""
CAN I WIN - STATE-BASED MINIMAX WITH BITMASKS:
==============================================

This problem demonstrates advanced Game Theory DP with complex state management:
- Bitmask representation of used numbers for efficient state encoding
- Minimax optimization with memoization over exponential state space
- Strategic analysis with immediate win detection and impossibility pruning
- Perfect information game with finite, deterministic outcomes

KEY INSIGHTS:
============
1. **BITMASK STATE ENCODING**: Efficiently represent used numbers with bit patterns
2. **EXPONENTIAL STATE SPACE**: 2^n possible states where n = maxChoosableInteger  
3. **MEMOIZATION CRITICAL**: Without caching, exponential time complexity is impractical
4. **IMMEDIATE WIN DETECTION**: Check for single-move victories before complex analysis
5. **FEASIBILITY PRUNING**: Early termination when desired total is impossible to reach

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(2^n × m) time, O(n) space
   - Pure recursive exploration without memoization
   - Exponential time complexity makes it impractical for large inputs

2. **Memoization**: O(2^n × m) time, O(2^n) space  
   - Top-down DP with state caching
   - Dramatically reduces repeated subproblem computation

3. **Optimized Memoization**: O(2^n) time, O(2^n) space
   - Eliminate current total from state (can be computed from bitmask)
   - Most efficient approach for this problem

4. **Strategy Analysis**: O(2^n) time, O(2^n) space
   - Include optimal move tracking and strategic insights
   - Essential for game AI and strategic understanding

CORE BITMASK MINIMAX ALGORITHM:
==============================
```python
def canIWin(maxChoosableInteger, desiredTotal):
    # Early termination checks
    if maxChoosableInteger >= desiredTotal:
        return True
    
    total_sum = maxChoosableInteger * (maxChoosableInteger + 1) // 2
    if total_sum < desiredTotal:
        return False
    
    memo = {}
    
    def canWin(used_mask):
        if used_mask in memo:
            return memo[used_mask]
        
        # Try each available number
        for num in range(1, maxChoosableInteger + 1):
            if used_mask & (1 << num):
                continue
            
            # Calculate current total
            current_total = sum(i for i in range(1, maxChoosableInteger + 1) 
                              if used_mask & (1 << i))
            
            # Immediate win check
            if current_total + num >= desiredTotal:
                memo[used_mask] = True
                return True
            
            # Strategic win check (opponent loses)
            new_mask = used_mask | (1 << num)
            if not canWin(new_mask):
                memo[used_mask] = True
                return True
        
        memo[used_mask] = False
        return False
    
    return canWin(0)
```

BITMASK STATE REPRESENTATION:
============================
**Encoding**: Each bit represents whether a number has been used
- `mask & (1 << i)` checks if number i is used
- `mask |= (1 << i)` marks number i as used
- Total of 2^n possible states for n choosable numbers

**State Compression**: Current total can be computed from bitmask, eliminating need for explicit tracking

**Memory Efficiency**: Compact representation enables memoization of large state spaces

GAME STATE ANALYSIS:
===================
**Immediate Win Conditions**: 
- `current_total + chosen_number >= desiredTotal`
- Can be checked before recursive exploration

**Strategic Win Conditions**:
- Choose number such that opponent has no winning response
- Requires recursive analysis of opponent's options

**Losing Positions**: 
- No available number leads to immediate or strategic win
- All opponent responses result in opponent victory

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(2^n) with memoization
- Each of 2^n states computed at most once
- Each state requires O(n) work to explore options

**Space Complexity**: O(2^n)
- Memoization table stores result for each possible state
- Practical limit: n ≤ 20 due to exponential space growth

**Optimization**: Current total elimination reduces constant factors

STRATEGIC INSIGHTS:
==================
**First Move Advantage**: First player gets first choice of optimal moves
**Critical Numbers**: Numbers that enable immediate victory from current state
**Impossible Games**: When sum of all numbers < desired total
**Trivial Games**: When maximum number ≥ desired total

EARLY TERMINATION CONDITIONS:
=============================
**Immediate Victory**: `maxChoosableInteger >= desiredTotal`
**Impossible Victory**: `sum(1..maxChoosableInteger) < desiredTotal`  
**Symmetry Breaking**: Identical states reached through different move sequences

APPLICATIONS:
============
- **Game AI**: Optimal strategy computation for number selection games
- **Decision Theory**: Competitive decision making with limited resources
- **Combinatorial Games**: Analysis of finite, perfect information games
- **Resource Allocation**: Strategic resource selection in competitive environments
- **Algorithm Design**: State-based optimization with exponential search spaces

RELATED PROBLEMS:
================
- **Nim Game**: Classic combinatorial game theory problem
- **Stone Games**: Resource selection with positional constraints  
- **Coin Games**: Similar mechanics with different winning conditions
- **Picking Numbers**: Various number selection game variants

VARIANTS:
========
- **Multi-Player**: Extension to more than two players
- **Number Reuse**: Allow repeated selection of same numbers
- **Range Restrictions**: Limit available number choices
- **Negative Numbers**: Include negative integers in selection pool

EDGE CASES:
==========
- **Single Number**: Trivial decision based on comparison
- **Impossible Total**: Early detection prevents unnecessary computation
- **Perfect Sum**: When all numbers exactly equal desired total
- **Large Ranges**: Exponential blowup for large maxChoosableInteger

OPTIMIZATION TECHNIQUES:
=======================
**State Compression**: Eliminate redundant state information
**Early Termination**: Quick resolution of trivial cases
**Memoization**: Essential for exponential state space management
**Bit Manipulation**: Efficient set operations using bitwise operators

This problem showcases how complex combinatorial games can be
solved optimally through systematic state space exploration,
demonstrating the power of memoization in taming exponential
complexity while maintaining optimal strategic analysis.
"""
