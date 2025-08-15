"""
LeetCode 1155: Number of Dice Rolls With Target Sum
Difficulty: Medium
Category: Knapsack Problems / Counting DP

PROBLEM DESCRIPTION:
===================
You have n dice, and each die has k faces numbered from 1 to k.

Given three integers n, k, and target, return the number of possible ways (out of the k^n total ways) 
to roll the dice, so the sum of the face-up numbers equals target.

Since the answer may be large, return it modulo 10^9 + 7.

Example 1:
Input: n = 1, k = 6, target = 3
Output: 1
Explanation: You throw one die with 6 faces.
There is only one way to get a sum of 3.

Example 2:
Input: n = 2, k = 6, target = 7
Output: 6
Explanation: You throw two dice, each with 6 faces.
There are 6 ways to get a sum of 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1).

Example 3:
Input: n = 30, k = 30, target = 500
Output: 222616187
Explanation: The answer must be returned modulo 10^9 + 7.

Constraints:
- 1 <= n, k <= 30
- 1 <= target <= 1000
"""

def num_rolls_to_target_bruteforce(n, k, target):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible dice combinations.
    
    Time Complexity: O(k^n) - k choices for each of n dice
    Space Complexity: O(n) - recursion stack depth
    """
    MOD = 10**9 + 7
    
    def count_ways(dice_left, current_target):
        if dice_left == 0:
            return 1 if current_target == 0 else 0
        
        if current_target <= 0:
            return 0
        
        ways = 0
        # Try all possible faces for current die
        for face in range(1, k + 1):
            ways += count_ways(dice_left - 1, current_target - face)
            ways %= MOD
        
        return ways
    
    return count_ways(n, target)


def num_rolls_to_target_memoization(n, k, target):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to cache subproblem results.
    
    Time Complexity: O(n * target * k) - states × transitions
    Space Complexity: O(n * target) - memoization table
    """
    MOD = 10**9 + 7
    memo = {}
    
    def count_ways(dice_left, current_target):
        if dice_left == 0:
            return 1 if current_target == 0 else 0
        
        if current_target <= 0:
            return 0
        
        if (dice_left, current_target) in memo:
            return memo[(dice_left, current_target)]
        
        ways = 0
        for face in range(1, k + 1):
            if current_target >= face:
                ways += count_ways(dice_left - 1, current_target - face)
                ways %= MOD
        
        memo[(dice_left, current_target)] = ways
        return ways
    
    return count_ways(n, target)


def num_rolls_to_target_dp_2d(n, k, target):
    """
    2D DYNAMIC PROGRAMMING:
    =======================
    Use 2D DP table for dice and target sum.
    
    Time Complexity: O(n * target * k) - fill DP table
    Space Complexity: O(n * target) - 2D DP table
    """
    MOD = 10**9 + 7
    
    # dp[i][j] = ways to get sum j using i dice
    dp = [[0] * (target + 1) for _ in range(n + 1)]
    
    # Base case: 0 dice, sum 0 = 1 way
    dp[0][0] = 1
    
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            # Try all possible faces for die i
            for face in range(1, k + 1):
                if j >= face:
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - face]) % MOD
    
    return dp[n][target]


def num_rolls_to_target_dp_1d(n, k, target):
    """
    1D DYNAMIC PROGRAMMING (SPACE OPTIMIZED):
    =========================================
    Use 1D DP array, process dice one by one.
    
    Time Complexity: O(n * target * k) - same iterations
    Space Complexity: O(target) - 1D DP array
    """
    MOD = 10**9 + 7
    
    # dp[j] = ways to get sum j with current number of dice
    dp = [0] * (target + 1)
    dp[0] = 1  # Base case: sum 0 with 0 dice
    
    for i in range(n):
        new_dp = [0] * (target + 1)
        
        for j in range(1, target + 1):
            # Try all possible faces for current die
            for face in range(1, k + 1):
                if j >= face:
                    new_dp[j] = (new_dp[j] + dp[j - face]) % MOD
        
        dp = new_dp
    
    return dp[target]


def num_rolls_to_target_optimized(n, k, target):
    """
    OPTIMIZED DP WITH EARLY TERMINATION:
    ===================================
    Add optimizations and early termination checks.
    
    Time Complexity: O(n * target * k) - worst case, often better
    Space Complexity: O(target) - 1D DP array
    """
    MOD = 10**9 + 7
    
    # Early termination checks
    if target < n or target > n * k:
        return 0  # Impossible to reach target
    
    if n == 1:
        return 1 if 1 <= target <= k else 0
    
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for i in range(n):
        new_dp = [0] * (target + 1)
        
        # Optimize range: only consider reachable sums
        min_sum = max(1, target - (n - i - 1) * k)  # Minimum possible with remaining dice
        max_sum = min(target, (i + 1) * k)          # Maximum possible with current dice
        
        for j in range(min_sum, max_sum + 1):
            for face in range(1, min(k, j) + 1):
                if j >= face:
                    new_dp[j] = (new_dp[j] + dp[j - face]) % MOD
        
        dp = new_dp
    
    return dp[target]


def num_rolls_to_target_rolling_sum(n, k, target):
    """
    ROLLING SUM OPTIMIZATION:
    ========================
    Use rolling sum to optimize inner loop.
    
    Time Complexity: O(n * target) - optimized inner loop
    Space Complexity: O(target) - DP array
    """
    MOD = 10**9 + 7
    
    if target < n or target > n * k:
        return 0
    
    dp = [0] * (target + 1)
    dp[0] = 1
    
    for i in range(n):
        new_dp = [0] * (target + 1)
        cumsum = 0
        
        for j in range(1, target + 1):
            # Add new element to sliding window
            if j <= k:
                cumsum = (cumsum + dp[j - 1]) % MOD
            
            # Remove old element from sliding window
            if j > k:
                cumsum = (cumsum - dp[j - k - 1] + MOD) % MOD
                cumsum = (cumsum + dp[j - 1]) % MOD
            
            new_dp[j] = cumsum
        
        dp = new_dp
    
    return dp[target]


def num_rolls_to_target_mathematical(n, k, target):
    """
    MATHEMATICAL ANALYSIS:
    =====================
    Analyze the problem with combinatorial insights.
    
    Time Complexity: O(n * target * k) - DP computation
    Space Complexity: O(target) - DP array
    """
    MOD = 10**9 + 7
    
    print(f"Mathematical Analysis:")
    print(f"n = {n} dice, k = {k} faces each, target = {target}")
    print(f"Total possible outcomes: {k}^{n} = {k**n}")
    print(f"Minimum possible sum: {n} (all 1s)")
    print(f"Maximum possible sum: {n * k} (all {k}s)")
    
    # Check feasibility
    if target < n or target > n * k:
        print(f"Impossible: target {target} outside range [{n}, {n * k}]")
        return 0
    
    # For small cases, show combinatorial insight
    if n <= 3 and k <= 6 and target <= 18:
        print(f"This is equivalent to finding coefficient of x^{target} in:")
        print(f"(x^1 + x^2 + ... + x^{k})^{n}")
        print(f"= (x(1-x^{k})/(1-x))^{n}")
    
    # Run optimized solution
    result = num_rolls_to_target_optimized(n, k, target)
    print(f"Number of ways: {result}")
    
    if k**n <= 1000000:  # For small total outcomes
        probability = result / (k**n)
        print(f"Probability: {result}/{k**n} = {probability:.6f}")
    
    return result


def num_rolls_to_target_with_sequences(n, k, target):
    """
    FIND ACTUAL SEQUENCES (FOR SMALL INPUTS):
    =========================================
    Generate actual dice sequences that sum to target.
    
    Time Complexity: O(k^n) - generate all sequences
    Space Complexity: O(k^n) - store sequences
    """
    if n > 10 or k > 6:  # Avoid explosion for large inputs
        return num_rolls_to_target_optimized(n, k, target), []
    
    def generate_sequences(dice_left, current_target, current_sequence):
        if dice_left == 0:
            if current_target == 0:
                return [current_sequence[:]]
            else:
                return []
        
        if current_target <= 0:
            return []
        
        sequences = []
        for face in range(1, k + 1):
            current_sequence.append(face)
            sequences.extend(generate_sequences(dice_left - 1, 
                                              current_target - face, 
                                              current_sequence))
            current_sequence.pop()
        
        return sequences
    
    all_sequences = generate_sequences(n, target, [])
    return len(all_sequences), all_sequences


def num_rolls_to_target_iterative(n, k, target):
    """
    ITERATIVE APPROACH:
    ==================
    Build solution iteratively without recursion.
    
    Time Complexity: O(n * target * k) - iterative loops
    Space Complexity: O(target) - DP array
    """
    MOD = 10**9 + 7
    
    if target < n or target > n * k:
        return 0
    
    # Start with 0 dice: only way to get sum 0 is with 0 dice
    current_ways = [0] * (target + 1)
    current_ways[0] = 1
    
    # Add dice one by one
    for dice in range(1, n + 1):
        next_ways = [0] * (target + 1)
        
        # For each possible sum with current number of dice
        for s in range(dice, min(target, dice * k) + 1):
            # For each possible face value
            for face in range(1, min(k, s) + 1):
                if s >= face:
                    next_ways[s] = (next_ways[s] + current_ways[s - face]) % MOD
        
        current_ways = next_ways
    
    return current_ways[target]


# Test cases
def test_num_rolls_to_target():
    """Test all implementations with various inputs"""
    test_cases = [
        (1, 6, 3, 1),
        (2, 6, 7, 6),
        (30, 30, 500, 222616187),
        (1, 2, 3, 0),  # Impossible
        (2, 2, 5, 0),  # Impossible
        (2, 2, 4, 1),  # Only (2,2)
        (3, 2, 3, 1),  # Only (1,1,1)
        (3, 2, 6, 1),  # Only (2,2,2)
        (2, 3, 4, 2),  # (1,3) and (3,1)
        (3, 3, 6, 10), # Multiple ways
        (4, 6, 10, 56),
        (2, 1, 2, 1),  # Only (1,1)
        (10, 10, 50, 16796160)
    ]
    
    print("Testing Number of Dice Rolls Solutions:")
    print("=" * 70)
    
    for i, (n, k, target, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n={n}, k={k}, target={target}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for large inputs)
        if k**n <= 100000:
            brute = num_rolls_to_target_bruteforce(n, k, target)
            print(f"Brute Force:      {brute:>8} {'✓' if brute == expected else '✗'}")
        
        memo = num_rolls_to_target_memoization(n, k, target)
        dp_2d = num_rolls_to_target_dp_2d(n, k, target)
        dp_1d = num_rolls_to_target_dp_1d(n, k, target)
        optimized = num_rolls_to_target_optimized(n, k, target)
        iterative = num_rolls_to_target_iterative(n, k, target)
        
        print(f"Memoization:      {memo:>8} {'✓' if memo == expected else '✗'}")
        print(f"2D DP:            {dp_2d:>8} {'✓' if dp_2d == expected else '✗'}")
        print(f"1D DP:            {dp_1d:>8} {'✓' if dp_1d == expected else '✗'}")
        print(f"Optimized:        {optimized:>8} {'✓' if optimized == expected else '✗'}")
        print(f"Iterative:        {iterative:>8} {'✓' if iterative == expected else '✗'}")
        
        # Show actual sequences for very small cases
        if expected > 0 and expected <= 10 and n <= 4:
            count, sequences = num_rolls_to_target_with_sequences(n, k, target)
            if sequences:
                print(f"Sequences: {sequences}")
    
    # Mathematical analysis example
    print(f"\n" + "=" * 70)
    print("MATHEMATICAL ANALYSIS EXAMPLE:")
    print("-" * 40)
    num_rolls_to_target_mathematical(2, 6, 7)
    
    print(f"\n" + "=" * 70)
    print("COMBINATORIAL INSIGHT:")
    print("This problem finds the coefficient of x^target in (x^1 + x^2 + ... + x^k)^n")
    print("Equivalent to distributing 'target' identical items into 'n' distinct groups")
    print("where each group gets between 1 and k items.")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(k^n),         Space: O(n)")
    print("Memoization:      Time: O(n*target*k),  Space: O(n*target)")
    print("2D DP:            Time: O(n*target*k),  Space: O(n*target)")
    print("1D DP:            Time: O(n*target*k),  Space: O(target)")
    print("Optimized:        Time: O(n*target*k),  Space: O(target)")
    print("Rolling Sum:      Time: O(n*target),    Space: O(target)")
    print("Iterative:        Time: O(n*target*k),  Space: O(target)")


if __name__ == "__main__":
    test_num_rolls_to_target()


"""
PATTERN RECOGNITION:
==================
This is a Counting DP problem with constraints:
- n identical dice, each with k faces (1 to k)
- Count ways to achieve exactly target sum
- Classic "coefficient extraction" from generating functions
- Similar to coin change counting, but with bounds on each "coin" use

KEY INSIGHT - GENERATING FUNCTION:
=================================
This problem asks for the coefficient of x^target in:
(x^1 + x^2 + ... + x^k)^n

Mathematical transformation:
- Each die contributes x^face_value
- n dice → product of n identical polynomials
- Target sum → find coefficient of x^target

STATE DEFINITION:
================
dp[i][j] = number of ways to get sum j using i dice
Or optimized: dp[j] = number of ways to get sum j (current number of dice)

RECURRENCE RELATION:
===================
dp[i][j] = Σ(dp[i-1][j-face]) for face in [1, k] where j >= face

Base case: dp[0][0] = 1 (one way to get sum 0 with 0 dice)

ALGORITHM PROGRESSION:
=====================
1. **Brute Force**: O(k^n) - try all k^n combinations
2. **Memoization**: O(n×target×k) - cache (dice_left, current_target)
3. **2D DP**: O(n×target×k) - tabulation approach
4. **1D DP**: O(n×target×k) time, O(target) space - space optimized
5. **Rolling Sum**: O(n×target) - optimize inner loop with sliding window

OPTIMIZATION TECHNIQUES:
=======================
1. **Space Optimization**: 2D → 1D DP array
2. **Range Optimization**: Only consider reachable sums
3. **Early Termination**: Check impossible cases (target < n or target > n×k)
4. **Rolling Sum**: Use sliding window for inner summation
5. **Boundary Optimization**: Tighten loop bounds

MATHEMATICAL PROPERTIES:
=======================
1. **Symmetry**: Same structure as polynomial coefficient extraction
2. **Convolution**: Each die adds convolution with uniform distribution
3. **Central Limit**: For large n, approaches normal distribution
4. **Bounds**: Always n ≤ target ≤ n×k for valid solutions

COMPARISON WITH SIMILAR PROBLEMS:
================================
- **Coin Change 2 (518)**: Unlimited coins, count combinations
- **Combination Sum IV (377)**: Count permutations with repetition
- **This Problem**: Fixed number of bounded choices

EDGE CASES:
==========
1. **n = 1**: Simple range check [1, k]
2. **target < n**: Impossible (minimum sum is n)
3. **target > n×k**: Impossible (maximum sum is n×k)
4. **k = 1**: Only one way if target = n, else 0

ROLLING SUM OPTIMIZATION:
========================
Instead of inner loop Σ(dp[j-face]) for face in [1,k]:
Use sliding window sum to achieve O(1) per position.

VARIANTS TO PRACTICE:
====================
- Coin Change variants (322, 518) - similar DP structure
- Combination Sum problems - counting with constraints
- Partition functions - mathematical background
- Generating functions - theoretical foundation

INTERVIEW TIPS:
==============
1. Recognize as counting DP with bounded choices
2. Start with recursive brute force approach
3. Add memoization with 2D state (dice, target)
4. Show 2D DP tabulation approach
5. **Critical**: Space optimize to 1D array
6. Explain early termination checks (impossible cases)
7. Mention rolling sum optimization for advanced
8. Discuss generating function interpretation
9. Handle modular arithmetic correctly (10^9 + 7)
10. Consider range optimizations for large inputs
"""
