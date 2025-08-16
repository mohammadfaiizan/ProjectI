"""
LeetCode 1444: Number of Ways of Cutting a Pizza
Difficulty: Hard
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given a rectangular pizza represented as a rows x cols matrix containing the following characters:
- 'A' (an apple)
- '.' (empty cell)

You have to cut the pizza into k pieces using k-1 cuts. For each cut you choose the direction (horizontal or vertical), for each piece of pizza at least one apple should remain.

Return the number of ways of cutting the pizza such that each piece contains at least one apple. Since the answer can be a huge number, return this modulo 10^9 + 7.

Example 1:
Input: pizza = ["A..", "AAA", "..."], k = 3
Output: 3
Explanation: The figure above shows the three ways to cut the pizza. Note that pieces of pizza should contain at least one apple.

Example 2:
Input: pizza = ["A..", "AA.", "..."], k = 3
Output: 1

Example 3:
Input: pizza = ["A..", "A..", "..."], k = 1
Output: 1

Constraints:
- 1 <= rows, cols <= 50
- rows == pizza.length
- cols == pizza[i].length
- 1 <= k <= 10
- pizza consists of characters 'A' and '.' only.
"""

def ways_to_cut_pizza_brute_force(pizza, k):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible ways to make k-1 cuts recursively.
    
    Time Complexity: O(4^k * m * n) - exponential in cuts and linear scan for apples
    Space Complexity: O(k) - recursion depth
    """
    MOD = 10**9 + 7
    rows, cols = len(pizza), len(pizza[0])
    
    def has_apple(r1, c1, r2, c2):
        """Check if rectangle has at least one apple"""
        for i in range(r1, r2):
            for j in range(c1, c2):
                if pizza[i][j] == 'A':
                    return True
        return False
    
    def dfs(r1, c1, r2, c2, cuts_left):
        # Base case: no more cuts needed
        if cuts_left == 0:
            return 1 if has_apple(r1, c1, r2, c2) else 0
        
        ways = 0
        
        # Try horizontal cuts
        for cut_row in range(r1 + 1, r2):
            # Top piece: (r1, c1) to (cut_row, c2)
            # Bottom piece: (cut_row, c1) to (r2, c2)
            if has_apple(r1, c1, cut_row, c2):
                ways += dfs(cut_row, c1, r2, c2, cuts_left - 1)
                ways %= MOD
        
        # Try vertical cuts
        for cut_col in range(c1 + 1, c2):
            # Left piece: (r1, c1) to (r2, cut_col)
            # Right piece: (r1, cut_col) to (r2, c2)
            if has_apple(r1, c1, r2, cut_col):
                ways += dfs(r1, cut_col, r2, c2, cuts_left - 1)
                ways %= MOD
        
        return ways
    
    return dfs(0, 0, rows, cols, k - 1)


def ways_to_cut_pizza_memoization(pizza, k):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(m^2 * n^2 * k) - each state computed once
    Space Complexity: O(m^2 * n^2 * k) - memoization table
    """
    MOD = 10**9 + 7
    rows, cols = len(pizza), len(pizza[0])
    memo = {}
    
    def has_apple(r1, c1, r2, c2):
        """Check if rectangle has at least one apple"""
        for i in range(r1, r2):
            for j in range(c1, c2):
                if pizza[i][j] == 'A':
                    return True
        return False
    
    def dfs(r1, c1, r2, c2, cuts_left):
        if (r1, c1, r2, c2, cuts_left) in memo:
            return memo[(r1, c1, r2, c2, cuts_left)]
        
        # Base case: no more cuts needed
        if cuts_left == 0:
            result = 1 if has_apple(r1, c1, r2, c2) else 0
            memo[(r1, c1, r2, c2, cuts_left)] = result
            return result
        
        ways = 0
        
        # Try horizontal cuts
        for cut_row in range(r1 + 1, r2):
            if has_apple(r1, c1, cut_row, c2):
                ways += dfs(cut_row, c1, r2, c2, cuts_left - 1)
                ways %= MOD
        
        # Try vertical cuts
        for cut_col in range(c1 + 1, c2):
            if has_apple(r1, c1, r2, cut_col):
                ways += dfs(r1, cut_col, r2, c2, cuts_left - 1)
                ways %= MOD
        
        memo[(r1, c1, r2, c2, cuts_left)] = ways
        return ways
    
    return dfs(0, 0, rows, cols, k - 1)


def ways_to_cut_pizza_prefix_sum_optimization(pizza, k):
    """
    PREFIX SUM OPTIMIZATION:
    =======================
    Use 2D prefix sums to check apple existence in O(1).
    
    Time Complexity: O(m^2 * n^2 * k) - but with O(1) apple checks
    Space Complexity: O(m * n + m^2 * n^2 * k) - prefix sums + memoization
    """
    MOD = 10**9 + 7
    rows, cols = len(pizza), len(pizza[0])
    
    # Build 2D prefix sum for apple counts
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            apple_count = 1 if pizza[i-1][j-1] == 'A' else 0
            prefix[i][j] = (apple_count + 
                          prefix[i-1][j] + 
                          prefix[i][j-1] - 
                          prefix[i-1][j-1])
    
    def has_apple_fast(r1, c1, r2, c2):
        """Check if rectangle has at least one apple using prefix sums"""
        apple_count = (prefix[r2][c2] - 
                      prefix[r1][c2] - 
                      prefix[r2][c1] + 
                      prefix[r1][c1])
        return apple_count > 0
    
    memo = {}
    
    def dfs(r1, c1, r2, c2, cuts_left):
        if (r1, c1, r2, c2, cuts_left) in memo:
            return memo[(r1, c1, r2, c2, cuts_left)]
        
        # Base case: no more cuts needed
        if cuts_left == 0:
            result = 1 if has_apple_fast(r1, c1, r2, c2) else 0
            memo[(r1, c1, r2, c2, cuts_left)] = result
            return result
        
        ways = 0
        
        # Try horizontal cuts
        for cut_row in range(r1 + 1, r2):
            if has_apple_fast(r1, c1, cut_row, c2):
                ways += dfs(cut_row, c1, r2, c2, cuts_left - 1)
                ways %= MOD
        
        # Try vertical cuts
        for cut_col in range(c1 + 1, c2):
            if has_apple_fast(r1, c1, r2, cut_col):
                ways += dfs(r1, cut_col, r2, c2, cuts_left - 1)
                ways %= MOD
        
        memo[(r1, c1, r2, c2, cuts_left)] = ways
        return ways
    
    return dfs(0, 0, rows, cols, k - 1)


def ways_to_cut_pizza_dp_bottom_up(pizza, k):
    """
    BOTTOM-UP DP APPROACH:
    =====================
    Build solution iteratively from smaller pieces to larger pieces.
    
    Time Complexity: O(m^2 * n^2 * k) - fill DP table
    Space Complexity: O(m * n * k) - DP table
    """
    MOD = 10**9 + 7
    rows, cols = len(pizza), len(pizza[0])
    
    # Build 2D prefix sum for apple counts
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            apple_count = 1 if pizza[i-1][j-1] == 'A' else 0
            prefix[i][j] = (apple_count + 
                          prefix[i-1][j] + 
                          prefix[i][j-1] - 
                          prefix[i-1][j-1])
    
    def has_apple(r1, c1, r2, c2):
        """Check if rectangle has at least one apple"""
        apple_count = (prefix[r2][c2] - 
                      prefix[r1][c2] - 
                      prefix[r2][c1] + 
                      prefix[r1][c1])
        return apple_count > 0
    
    # dp[r][c][pieces] = ways to cut pizza starting from (r,c) into 'pieces' pieces
    dp = [[[0] * (k + 1) for _ in range(cols + 1)] for _ in range(rows + 1)]
    
    # Base case: 1 piece (no cuts needed)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if r < rows and c < cols and has_apple(r, c, rows, cols):
                dp[r][c][1] = 1
    
    # Fill DP table for increasing number of pieces
    for pieces in range(2, k + 1):
        for r in range(rows):
            for c in range(cols):
                # Try horizontal cuts
                for cut_row in range(r + 1, rows):
                    if has_apple(r, c, cut_row, cols):
                        dp[r][c][pieces] += dp[cut_row][c][pieces - 1]
                        dp[r][c][pieces] %= MOD
                
                # Try vertical cuts
                for cut_col in range(c + 1, cols):
                    if has_apple(r, c, rows, cut_col):
                        dp[r][c][pieces] += dp[r][cut_col][pieces - 1]
                        dp[r][c][pieces] %= MOD
    
    return dp[0][0][k]


def ways_to_cut_pizza_optimized(pizza, k):
    """
    MOST OPTIMIZED APPROACH:
    =======================
    Combine all optimizations: prefix sums + efficient DP state representation.
    
    Time Complexity: O(m^2 * n^2 * k) - optimal for this problem
    Space Complexity: O(m * n * k) - optimal space usage
    """
    MOD = 10**9 + 7
    rows, cols = len(pizza), len(pizza[0])
    
    # Precompute apple counts for all possible rectangles
    # apples[r][c] = number of apples in rectangle from (r,c) to (rows-1, cols-1)
    apples = [[0] * cols for _ in range(rows)]
    
    # Build suffix sum (bottom-right to top-left)
    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            apples[r][c] = (
                (1 if pizza[r][c] == 'A' else 0) +
                (apples[r+1][c] if r+1 < rows else 0) +
                (apples[r][c+1] if c+1 < cols else 0) -
                (apples[r+1][c+1] if r+1 < rows and c+1 < cols else 0)
            )
    
    # Memoization for DP
    memo = {}
    
    def dp(r, c, pieces_left):
        if pieces_left == 1:
            # Only one piece left, check if it has apples
            return 1 if apples[r][c] > 0 else 0
        
        if (r, c, pieces_left) in memo:
            return memo[(r, c, pieces_left)]
        
        ways = 0
        
        # Try horizontal cuts
        for cut_row in range(r + 1, rows):
            # Top piece: (r, c) to (cut_row-1, cols-1)
            top_apples = apples[r][c] - apples[cut_row][c]
            if top_apples > 0:
                ways += dp(cut_row, c, pieces_left - 1)
                ways %= MOD
        
        # Try vertical cuts
        for cut_col in range(c + 1, cols):
            # Left piece: (r, c) to (rows-1, cut_col-1)
            left_apples = apples[r][c] - apples[r][cut_col]
            if left_apples > 0:
                ways += dp(r, cut_col, pieces_left - 1)
                ways %= MOD
        
        memo[(r, c, pieces_left)] = ways
        return ways
    
    return dp(0, 0, k)


def ways_to_cut_pizza_with_visualization(pizza, k):
    """
    VISUALIZATION AND ANALYSIS:
    ==========================
    Show cutting process and analyze different cutting strategies.
    """
    MOD = 10**9 + 7
    rows, cols = len(pizza), len(pizza[0])
    
    print(f"Pizza Layout ({rows}x{cols}):")
    for i, row in enumerate(pizza):
        print(f"  Row {i}: {row}")
    print(f"Need to cut into {k} pieces")
    
    # Build apple position map
    apple_positions = []
    for r in range(rows):
        for c in range(cols):
            if pizza[r][c] == 'A':
                apple_positions.append((r, c))
    
    print(f"\nApple positions: {apple_positions}")
    print(f"Total apples: {len(apple_positions)}")
    
    # Build suffix sum for analysis
    apples = [[0] * cols for _ in range(rows)]
    
    for r in range(rows - 1, -1, -1):
        for c in range(cols - 1, -1, -1):
            apples[r][c] = (
                (1 if pizza[r][c] == 'A' else 0) +
                (apples[r+1][c] if r+1 < rows else 0) +
                (apples[r][c+1] if c+1 < cols else 0) -
                (apples[r+1][c+1] if r+1 < rows and c+1 < cols else 0)
            )
    
    print(f"\nApple count matrix (suffix sums):")
    for r in range(rows):
        row_str = "  "
        for c in range(cols):
            row_str += f"{apples[r][c]:3}"
        print(row_str)
    
    # Analyze cutting possibilities
    print(f"\nCutting analysis for first cut:")
    
    # Horizontal cuts
    print("Horizontal cuts:")
    for cut_row in range(1, rows):
        top_apples = apples[0][0] - apples[cut_row][0]
        bottom_apples = apples[cut_row][0]
        print(f"  Cut after row {cut_row-1}: top={top_apples}, bottom={bottom_apples}")
    
    # Vertical cuts
    print("Vertical cuts:")
    for cut_col in range(1, cols):
        left_apples = apples[0][0] - apples[0][cut_col]
        right_apples = apples[0][cut_col]
        print(f"  Cut after col {cut_col-1}: left={left_apples}, right={right_apples}")
    
    # Calculate result
    result = ways_to_cut_pizza_optimized(pizza, k)
    print(f"\nTotal ways to cut pizza into {k} pieces: {result}")
    
    return result


# Test cases
def test_ways_to_cut_pizza():
    """Test all implementations with various inputs"""
    test_cases = [
        (["A..", "AAA", "..."], 3, 3),
        (["A..", "AA.", "..."], 3, 1),
        (["A..", "A..", "..."], 1, 1),
        (["AAA", "AAA", "AAA"], 3, 12),
        (["A"], 1, 1),
        (["...", "...", "..."], 1, 0),  # No apples
        (["A.", ".A"], 2, 1),
        (["AA", "AA"], 2, 3),
        (["A.A", ".A.", "A.A"], 3, 2),
        (["AAA"], 3, 1)
    ]
    
    print("Testing Ways to Cut Pizza Solutions:")
    print("=" * 70)
    
    for i, (pizza, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Pizza: {pizza}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(pizza) <= 3 and len(pizza[0]) <= 3 and k <= 3:
            try:
                brute = ways_to_cut_pizza_brute_force(pizza, k)
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Error")
        
        memo_result = ways_to_cut_pizza_memoization(pizza, k)
        prefix_result = ways_to_cut_pizza_prefix_sum_optimization(pizza, k)
        dp_result = ways_to_cut_pizza_dp_bottom_up(pizza, k)
        optimized_result = ways_to_cut_pizza_optimized(pizza, k)
        
        print(f"Memoization:      {memo_result:>5} {'✓' if memo_result == expected else '✗'}")
        print(f"Prefix Sum:       {prefix_result:>5} {'✓' if prefix_result == expected else '✗'}")
        print(f"DP Bottom-up:     {dp_result:>5} {'✓' if dp_result == expected else '✗'}")
        print(f"Optimized:        {optimized_result:>5} {'✓' if optimized_result == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    ways_to_cut_pizza_with_visualization(["A..", "AAA", "..."], 3)
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. RECURSIVE STRUCTURE: Each cut creates smaller subproblems")
    print("2. APPLE CONSTRAINT: Every piece must contain at least one apple")
    print("3. CUT DIRECTIONS: Both horizontal and vertical cuts possible")
    print("4. SUFFIX SUMS: Efficient apple counting for remaining rectangles")
    print("5. MEMOIZATION: Avoid recomputing same (position, pieces) states")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all possible cutting sequences")
    print("Memoization:      Cache results for (position, pieces) states")
    print("Prefix Sum:       O(1) apple existence checks")
    print("DP Bottom-up:     Iterative DP building from base cases")
    print("Optimized:        Suffix sums + efficient state representation")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(4^k * mn),     Space: O(k)")
    print("Memoization:      Time: O(m²n²k),       Space: O(m²n²k)")
    print("Prefix Sum:       Time: O(m²n²k),       Space: O(mn + m²n²k)")
    print("DP Bottom-up:     Time: O(m²n²k),       Space: O(mnk)")
    print("Optimized:        Time: O(m²n²k),       Space: O(mn + mnk)")


if __name__ == "__main__":
    test_ways_to_cut_pizza()


"""
PATTERN RECOGNITION:
==================
Pizza Cutting is a combinatorial optimization problem with constraints:
- Recursive structure where each cut creates independent subproblems
- Constraint satisfaction (each piece must have at least one apple)
- 2D geometric cutting with directional choices
- Demonstrates advanced memoization and state space optimization

KEY INSIGHT - CONSTRAINED RECURSIVE CUTTING:
============================================
**Problem Structure**: 
- Make k-1 cuts to create k pieces
- Each cut can be horizontal or vertical
- Every resulting piece must contain at least one apple

**State Representation**: (top_row, left_col, pieces_remaining)
- Represents cutting remaining rectangle into pieces_remaining pieces
- Rectangle goes from (top_row, left_col) to (rows-1, cols-1)

**Transition**: For each possible cut position:
- Check if cut piece has apples
- Recursively solve for remaining rectangle with pieces_remaining-1

ALGORITHM APPROACHES:
====================

1. **Brute Force Recursion**: O(4^k × mn) time
   - Try all possible cut sequences
   - Linear scan to check apple existence
   - Exponential in number of cuts

2. **Memoization**: O(m²n²k) time, O(m²n²k) space
   - Cache results for (r, c, pieces) states
   - Avoid recomputing same subproblems
   - Still linear scan for apples

3. **Prefix Sum Optimization**: O(m²n²k) time, O(mn + m²n²k) space
   - O(1) apple existence checks using 2D prefix sums
   - Same time complexity but better constants
   - Additional space for prefix sum matrix

4. **Optimized with Suffix Sums**: O(m²n²k) time, O(mn + mnk) space
   - Use suffix sums from bottom-right
   - More efficient state representation
   - Optimal space complexity

SUFFIX SUM TECHNIQUE:
====================
**Key Insight**: Instead of checking arbitrary rectangles, only check rectangles 
that extend from current position to bottom-right corner.

```
apples[r][c] = number of apples in rectangle from (r,c) to (rows-1, cols-1)

Build using suffix sum:
apples[r][c] = pizza[r][c] + apples[r+1][c] + apples[r][c+1] - apples[r+1][c+1]
```

**Cut Analysis**:
- Horizontal cut at row h: top piece has apples[r][c] - apples[h][c] apples
- Vertical cut at col v: left piece has apples[r][c] - apples[r][v] apples

STATE SPACE OPTIMIZATION:
=========================
**Original State**: (top_row, left_col, bottom_row, right_col, pieces_remaining)
- O(m²n²k) states, each with O(m+n) transitions

**Optimized State**: (top_row, left_col, pieces_remaining)
- O(mnk) states, each with O(m+n) transitions
- Assumes remaining rectangle extends to bottom-right

**Memory Layout**:
```python
memo = {}  # (r, c, pieces) -> ways

def dp(r, c, pieces):
    if (r, c, pieces) in memo:
        return memo[(r, c, pieces)]
    # ... computation ...
    memo[(r, c, pieces)] = result
    return result
```

CUTTING STRATEGY ANALYSIS:
=========================
**Valid Cut Conditions**:
1. Cut must create non-empty pieces
2. Separated piece must contain at least one apple
3. Remaining piece must be processable for remaining cuts

**Cut Enumeration**:
```python
# Horizontal cuts
for cut_row in range(r + 1, rows):
    if top_piece_has_apples:
        ways += dp(cut_row, c, pieces - 1)

# Vertical cuts  
for cut_col in range(c + 1, cols):
    if left_piece_has_apples:
        ways += dp(r, cut_col, pieces - 1)
```

APPLICATIONS:
============
1. **Manufacturing**: Cutting materials with quality constraints
2. **Image Processing**: Segmenting images with feature requirements
3. **Resource Division**: Partitioning areas with minimum resource constraints
4. **Game Development**: Level design with objective placement

VARIANTS TO PRACTICE:
====================
- Minimum Cost to Cut a Stick (1547) - similar cutting optimization
- Burst Balloons (312) - interval DP with optimal cutting
- Matrix Chain Multiplication (classic) - optimal parenthesization
- Palindrome Partitioning II (132) - string cutting with constraints

INTERVIEW TIPS:
==============
1. **Identify recursive structure**: Each cut creates independent subproblems
2. **State representation**: What information needed to solve subproblem?
3. **Constraint handling**: How to efficiently check apple requirements?
4. **Optimization techniques**: Prefix/suffix sums for range queries
5. **Space optimization**: Reduce state dimensions when possible
6. **Base cases**: Handle single piece (no cuts) correctly
7. **Modular arithmetic**: Large results need modulo operations
8. **Edge cases**: No apples, single cell, k=1
9. **Real applications**: Manufacturing, image processing
10. **Alternative formulations**: Different cutting rules, multiple constraints

MATHEMATICAL INSIGHT:
====================
This problem demonstrates **constrained combinatorial enumeration**:
- **Recursive decomposition** breaks complex cutting into simpler subproblems
- **Constraint propagation** ensures every piece meets requirements
- **Dynamic programming** eliminates redundant computation through memoization

The combination of **geometric partitioning** with **constraint satisfaction** 
creates a rich problem space that requires both algorithmic sophistication 
and efficient implementation techniques.

The problem showcases how **problem-specific optimizations** (suffix sums, 
state space reduction) can significantly improve both time and space complexity 
while maintaining correctness.
"""
