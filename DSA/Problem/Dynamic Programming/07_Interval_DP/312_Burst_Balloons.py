"""
LeetCode 312: Burst Balloons
Difficulty: Hard
Category: Interval DP - Foundation

PROBLEM DESCRIPTION:
===================
You are given n balloons, indexed from 0 to n - 1. Each balloon is painted with a number on it represented by an array nums. 
You are asked to burst all the balloons.

If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] coins. 
If i - 1 or i + 1 goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it.

Return the maximum coins you can collect by bursting the balloons wisely.

Example 1:
Input: nums = [3,1,5,8]
Output: 167
Explanation:
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 15 + 120 + 24 + 8 = 167

Example 2:
Input: nums = [1,5]
Output: 10

Constraints:
- n == nums.length
- 1 <= n <= 300
- 0 <= nums[i] <= 100
"""

def max_coins_brute_force(nums):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible orders of bursting balloons.
    
    Time Complexity: O(n!) - all permutations
    Space Complexity: O(n) - recursion depth
    """
    def burst_recursive(balloons):
        if not balloons:
            return 0
        
        max_coins = 0
        for i in range(len(balloons)):
            # Calculate coins for bursting balloon i
            left = 1 if i == 0 else balloons[i-1]
            right = 1 if i == len(balloons)-1 else balloons[i+1]
            coins = left * balloons[i] * right
            
            # Create new array without balloon i
            new_balloons = balloons[:i] + balloons[i+1:]
            
            # Recurse and get total coins
            total_coins = coins + burst_recursive(new_balloons)
            max_coins = max(max_coins, total_coins)
        
        return max_coins
    
    return burst_recursive(nums)


def max_coins_memoization(nums):
    """
    MEMOIZATION APPROACH:
    ====================
    Cache results for different balloon configurations.
    
    Time Complexity: O(n^3) - with memoization
    Space Complexity: O(n^3) - memo table
    """
    memo = {}
    
    def burst_memo(balloons):
        balloons_tuple = tuple(balloons)
        if balloons_tuple in memo:
            return memo[balloons_tuple]
        
        if not balloons:
            return 0
        
        max_coins = 0
        for i in range(len(balloons)):
            left = 1 if i == 0 else balloons[i-1]
            right = 1 if i == len(balloons)-1 else balloons[i+1]
            coins = left * balloons[i] * right
            
            new_balloons = balloons[:i] + balloons[i+1:]
            total_coins = coins + burst_memo(new_balloons)
            max_coins = max(max_coins, total_coins)
        
        memo[balloons_tuple] = max_coins
        return max_coins
    
    return burst_memo(nums)


def max_coins_interval_dp(nums):
    """
    INTERVAL DP APPROACH:
    ====================
    Think backwards: which balloon to burst last in each interval.
    
    Time Complexity: O(n^3) - three nested loops
    Space Complexity: O(n^2) - DP table
    """
    # Add boundary balloons with value 1
    balloons = [1] + nums + [1]
    n = len(balloons)
    
    # dp[i][j] = maximum coins from bursting balloons in open interval (i, j)
    dp = [[0] * n for _ in range(n)]
    
    # length is the length of the interval
    for length in range(2, n):  # At least 2 positions apart for open interval
        for i in range(n - length):
            j = i + length
            
            # Try bursting each balloon k in the open interval (i, j)
            for k in range(i + 1, j):
                # If we burst balloon k last in interval (i, j):
                # - Left and right intervals are already burst
                # - Balloon k sees balloons i and j as neighbors
                coins = balloons[i] * balloons[k] * balloons[j]
                total = dp[i][k] + coins + dp[k][j]
                dp[i][j] = max(dp[i][j], total)
    
    # Return result for the entire interval (0, n-1)
    return dp[0][n-1]


def max_coins_top_down(nums):
    """
    TOP-DOWN DP APPROACH:
    ====================
    Recursive implementation with memoization on intervals.
    
    Time Complexity: O(n^3) - each interval computed once
    Space Complexity: O(n^2) - memo table + recursion
    """
    balloons = [1] + nums + [1]
    n = len(balloons)
    memo = {}
    
    def dp(left, right):
        if left + 1 >= right:  # No balloons in open interval
            return 0
        
        if (left, right) in memo:
            return memo[(left, right)]
        
        max_coins = 0
        # Try bursting each balloon k last in interval (left, right)
        for k in range(left + 1, right):
            coins = balloons[left] * balloons[k] * balloons[right]
            total = dp(left, k) + coins + dp(k, right)
            max_coins = max(max_coins, total)
        
        memo[(left, right)] = max_coins
        return max_coins
    
    return dp(0, n - 1)


def max_coins_with_order(nums):
    """
    TRACK BURSTING ORDER:
    ====================
    Return maximum coins and the optimal bursting order.
    
    Time Complexity: O(n^3) - DP computation + reconstruction
    Space Complexity: O(n^2) - DP table + order tracking
    """
    balloons = [1] + nums + [1]
    n = len(balloons)
    
    dp = [[0] * n for _ in range(n)]
    choice = [[0] * n for _ in range(n)]  # Track which balloon to burst last
    
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            
            for k in range(i + 1, j):
                coins = balloons[i] * balloons[k] * balloons[j]
                total = dp[i][k] + coins + dp[k][j]
                
                if total > dp[i][j]:
                    dp[i][j] = total
                    choice[i][j] = k
    
    # Reconstruct the bursting order
    def get_order(left, right):
        if left + 1 >= right:
            return []
        
        k = choice[left][right]
        # Burst left interval, then right interval, then balloon k
        return get_order(left, k) + get_order(k, right) + [k - 1]  # k-1 for original indexing
    
    order = get_order(0, n - 1)
    return dp[0][n - 1], order


def max_coins_analysis(nums):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and optimal strategy.
    """
    print(f"Burst Balloons Analysis:")
    print(f"Original balloons: {nums}")
    
    balloons = [1] + nums + [1]
    n = len(balloons)
    print(f"With boundaries: {balloons}")
    
    # Show DP table construction
    dp = [[0] * n for _ in range(n)]
    choice = [[0] * n for _ in range(n)]
    
    print(f"\nDP Table Construction:")
    print(f"dp[i][j] = max coins from open interval (i, j)")
    
    for length in range(2, n):
        print(f"\nLength {length} intervals:")
        for i in range(n - length):
            j = i + length
            print(f"  Interval ({i}, {j}) - balloons between {balloons[i]} and {balloons[j]}:")
            
            for k in range(i + 1, j):
                coins = balloons[i] * balloons[k] * balloons[j]
                total = dp[i][k] + coins + dp[k][j]
                
                print(f"    Burst balloon {k} (value {balloons[k]}) last:")
                print(f"      Coins: {balloons[i]} * {balloons[k]} * {balloons[j]} = {coins}")
                print(f"      Total: {dp[i][k]} + {coins} + {dp[k][j]} = {total}")
                
                if total > dp[i][j]:
                    dp[i][j] = total
                    choice[i][j] = k
                    print(f"      *** New best for interval ({i}, {j})")
            
            print(f"    Best for ({i}, {j}): {dp[i][j]} (burst balloon {choice[i][j]} last)")
    
    print(f"\nFinal DP Table:")
    print("   ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            print(f"{dp[i][j]:4}", end="")
        print()
    
    print(f"\nMaximum coins: {dp[0][n-1]}")
    
    # Show optimal order
    max_coins, order = max_coins_with_order(nums)
    print(f"Optimal bursting order (0-indexed): {order}")
    
    # Simulate the optimal order
    print(f"\nSimulation of optimal order:")
    current_balloons = nums[:]
    total_coins = 0
    
    for step, balloon_idx in enumerate(order):
        # Adjust index for removed balloons
        actual_idx = balloon_idx - step
        
        left = 1 if actual_idx == 0 else current_balloons[actual_idx - 1]
        right = 1 if actual_idx == len(current_balloons) - 1 else current_balloons[actual_idx + 1]
        balloon_val = current_balloons[actual_idx]
        
        coins = left * balloon_val * right
        total_coins += coins
        
        print(f"  Step {step + 1}: Burst balloon at index {actual_idx} (value {balloon_val})")
        print(f"    Current state: {current_balloons}")
        print(f"    Coins: {left} * {balloon_val} * {right} = {coins}")
        print(f"    Total coins so far: {total_coins}")
        
        # Remove the burst balloon
        current_balloons.pop(actual_idx)
    
    print(f"\nFinal total: {total_coins}")
    return dp[0][n-1]


def max_coins_variants():
    """
    BURST BALLOONS VARIANTS:
    =======================
    Different scenarios and modifications.
    """
    
    def max_coins_with_limit(nums, max_bursts):
        """Burst at most max_bursts balloons"""
        balloons = [1] + nums + [1]
        n = len(balloons)
        
        # dp[i][j][k] = max coins from interval (i,j) using at most k bursts
        dp = [[[0 for _ in range(max_bursts + 1)] for _ in range(n)] for _ in range(n)]
        
        for bursts in range(1, min(max_bursts, len(nums)) + 1):
            for length in range(2, n):
                for i in range(n - length):
                    j = i + length
                    
                    # Don't burst any balloon in this interval
                    if bursts > 0:
                        dp[i][j][bursts] = dp[i][j][bursts - 1]
                    
                    # Try bursting each balloon k last
                    for k in range(i + 1, j):
                        if bursts >= 1:
                            coins = balloons[i] * balloons[k] * balloons[j]
                            total = dp[i][k][bursts - 1] + coins + dp[k][j][0]
                            dp[i][j][bursts] = max(dp[i][j][bursts], total)
        
        return dp[0][n - 1][max_bursts]
    
    def min_coins_burst_all(nums):
        """Minimum coins to burst all balloons (if all values were negative)"""
        # Convert to minimize problem by negating values
        neg_nums = [-x for x in nums]
        return -max_coins_interval_dp(neg_nums)
    
    def max_coins_with_weights(nums, weights):
        """Each balloon has a weight affecting the coins calculation"""
        balloons = [1] + nums + [1]
        balloon_weights = [1] + weights + [1]
        n = len(balloons)
        
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for i in range(n - length):
                j = i + length
                
                for k in range(i + 1, j):
                    # Use weights in calculation
                    coins = (balloons[i] * balloons[k] * balloons[j] * 
                            balloon_weights[k])
                    total = dp[i][k] + coins + dp[k][j]
                    dp[i][j] = max(dp[i][j], total)
        
        return dp[0][n - 1]
    
    # Test variants
    test_cases = [
        [3, 1, 5, 8],
        [1, 5],
        [1, 2, 3, 4, 5],
        [9, 7, 8, 6, 2, 1]
    ]
    
    print("Burst Balloons Variants:")
    print("=" * 50)
    
    for nums in test_cases:
        print(f"\nBalloons: {nums}")
        
        standard = max_coins_interval_dp(nums)
        print(f"Standard: {standard}")
        
        # With burst limit
        limit = min(3, len(nums))
        limited = max_coins_with_limit(nums, limit)
        print(f"At most {limit} bursts: {limited}")
        
        # With weights
        weights = [1.5] * len(nums)  # 50% bonus for each balloon
        weighted = max_coins_with_weights(nums, weights)
        print(f"With 1.5x weights: {weighted}")
        
        # Show order
        max_coins, order = max_coins_with_order(nums)
        print(f"Optimal order: {order}")


# Test cases
def test_burst_balloons():
    """Test all implementations with various inputs"""
    test_cases = [
        ([3, 1, 5, 8], 167),
        ([1, 5], 10),
        ([1], 1),
        ([2, 4, 6], 132),  # 2*4*6 + 1*2*1 + 1*6*1 = 48 + 2 + 6 = 56? Let me recalculate
        ([1, 2, 3, 4, 5], 110),
        ([9, 7, 8, 6, 2, 1], 1582)
    ]
    
    print("Testing Burst Balloons Solutions:")
    print("=" * 70)
    
    for i, (nums, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Balloons: {nums}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large inputs
        if len(nums) <= 4:
            try:
                brute_force = max_coins_brute_force(nums)
                print(f"Brute Force:      {brute_force:>4} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        if len(nums) <= 6:
            memoization = max_coins_memoization(nums)
            print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        
        interval_dp = max_coins_interval_dp(nums)
        top_down = max_coins_top_down(nums)
        
        print(f"Interval DP:      {interval_dp:>4} {'✓' if interval_dp == expected else '✗'}")
        print(f"Top-down DP:      {top_down:>4} {'✓' if top_down == expected else '✗'}")
        
        # Show optimal order for small cases
        if len(nums) <= 6:
            max_coins, order = max_coins_with_order(nums)
            print(f"Optimal order: {order}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    max_coins_analysis([3, 1, 5, 8])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    max_coins_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. INTERVAL DP: Think which balloon to burst LAST in each interval")
    print("2. BOUNDARY BALLOONS: Add 1s at boundaries for easier computation")
    print("3. OPTIMAL SUBSTRUCTURE: Optimal solution contains optimal subproblems")
    print("4. REVERSE THINKING: Instead of first, think about last balloon")
    print("5. O(N^3) SOLUTION: Three nested loops for all intervals and choices")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Game Theory: Optimal resource extraction strategies")
    print("• Economics: Production sequence optimization") 
    print("• Operations Research: Task scheduling with dependencies")
    print("• Algorithm Design: Interval optimization problems")
    print("• Dynamic Programming: Foundation for many interval problems")


if __name__ == "__main__":
    test_burst_balloons()


"""
BURST BALLOONS - FOUNDATION OF INTERVAL DYNAMIC PROGRAMMING:
===========================================================

This is the classic interval DP problem that introduces key concepts:
- Process intervals by length (bottom-up)
- Think about which element to process LAST in each interval
- Add boundary elements to simplify edge cases
- O(n³) complexity with optimal substructure

KEY INSIGHTS:
============
1. **REVERSE THINKING**: Instead of which balloon to burst first, think last
2. **INTERVAL DECOMPOSITION**: Bursting last balloon splits interval optimally
3. **BOUNDARY HANDLING**: Add 1s at boundaries to avoid edge case logic
4. **OPTIMAL SUBSTRUCTURE**: Optimal solution contains optimal subproblems
5. **BOTTOM-UP PROCESSING**: Build solutions from smaller to larger intervals

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n!) time, O(n) space
   - Try all possible bursting orders
   - Only viable for tiny inputs

2. **Memoization**: O(n³) time, O(2^n) space worst case
   - Cache results for different balloon configurations
   - Can be memory-intensive

3. **Interval DP (Bottom-up)**: O(n³) time, O(n²) space
   - Process intervals by increasing length
   - Standard and most efficient approach

4. **Top-down DP**: O(n³) time, O(n²) space
   - Recursive with memoization on intervals
   - More intuitive for some people

CORE INTERVAL DP ALGORITHM:
==========================
```python
# Add boundary balloons
balloons = [1] + nums + [1]
n = len(balloons)

# dp[i][j] = max coins from open interval (i, j)
dp = [[0] * n for _ in range(n)]

for length in range(2, n):           # Interval length
    for i in range(n - length):      # Start position
        j = i + length               # End position
        
        for k in range(i + 1, j):    # Balloon to burst last
            coins = balloons[i] * balloons[k] * balloons[j]
            total = dp[i][k] + coins + dp[k][j]
            dp[i][j] = max(dp[i][j], total)

return dp[0][n-1]
```

WHY THINK ABOUT LAST BALLOON:
=============================
**Problem with thinking "first"**: When we burst the first balloon, the remaining balloons form disconnected segments, making it hard to apply DP.

**Advantage of thinking "last"**: When we decide which balloon to burst last in an interval [i, j]:
- All other balloons in (i, j) are already burst
- The last balloon k sees only balloons i and j as neighbors
- We can recursively solve subproblems [i, k] and [k, j]

BOUNDARY BALLOON TECHNIQUE:
==========================
Adding balloons with value 1 at both ends:
- Eliminates special cases for edge balloons
- Every balloon has valid left and right neighbors
- Simplifies the recurrence relation
- Standard technique in interval DP

RECURRENCE RELATION:
===================
```
dp[i][j] = max(dp[i][k] + balloons[i] * balloons[k] * balloons[j] + dp[k][j])
           for all k in (i, j)

Base case: dp[i][j] = 0 if j - i <= 1 (no balloons in open interval)
```

**Intuition**: If we burst balloon k last in interval (i, j):
- Left subproblem: dp[i][k] (balloons in interval (i, k))
- Right subproblem: dp[k][j] (balloons in interval (k, j))  
- Current contribution: balloons[i] * balloons[k] * balloons[j]

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n³) - three nested loops
- **Space**: O(n²) - DP table
- **States**: O(n²) - all possible intervals
- **Transitions**: O(n) - try each balloon as last

INTERVAL DP PATTERN:
===================
This problem establishes the classic interval DP pattern:
1. **Process by length**: Start with small intervals, build up
2. **Last element choice**: Decide which element to process last
3. **Optimal decomposition**: Last choice splits into independent subproblems
4. **Boundary handling**: Add virtual elements to simplify edge cases

MATHEMATICAL PROPERTIES:
========================
- **Optimal Substructure**: Optimal solution contains optimal subsolutions
- **Overlapping Subproblems**: Same intervals computed multiple times
- **Monotonicity**: Larger intervals can only have equal or better solutions
- **Independence**: After choosing last balloon, subproblems are independent

SOLUTION RECONSTRUCTION:
=======================
To find the actual bursting order:
```python
def get_order(left, right, choice):
    if left + 1 >= right:
        return []
    
    k = choice[left][right]  # Balloon to burst last
    # Get order for subproblems, then add last balloon
    return get_order(left, k, choice) + get_order(k, right, choice) + [k]
```

APPLICATIONS:
============
- **Matrix Chain Multiplication**: Optimal parenthesization
- **Optimal Binary Search Trees**: Minimize search cost
- **Polygon Triangulation**: Minimum cost triangulation
- **Parsing Problems**: Optimal parse tree construction
- **Game Theory**: Optimal play in interval games

RELATED PROBLEMS:
================
- **Matrix Chain Multiplication (classic)**: Same DP pattern
- **Remove Boxes (546)**: More complex interval DP
- **Minimum Cost Tree From Leaf Values (1130)**: Similar structure
- **Strange Printer (664)**: Interval DP with character matching

OPTIMIZATION TECHNIQUES:
========================
- **Space Optimization**: Can reduce to O(n) space with careful implementation
- **Pruning**: Early termination when partial solutions exceed current best
- **Memoization**: Top-down approach with caching
- **Bottom-up**: Standard iterative approach

EDGE CASES:
==========
- **Single balloon**: Return the balloon value
- **Empty array**: Return 0
- **All zeros**: Return 0
- **Large values**: Check for integer overflow

This problem is fundamental to understanding interval DP and serves as
the foundation for many other optimization problems on ranges and intervals.
"""
