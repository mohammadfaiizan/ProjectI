"""
LeetCode 1467: Probability of a Two Balls Having The Same Color (Similar Pattern)
Extended Problem: Probability of Valid Sequence
Difficulty: Hard
Category: Probability DP - Combinatorial Probability

PROBLEM DESCRIPTION:
===================
Given n pairs of balls where each pair has a different color, we randomly distribute these 2n balls into two boxes such that each box contains exactly n balls.

What is the probability that both boxes contain the same number of balls of each color?

Extended to general probability sequence problems:
- Calculate probability of specific arrangements
- Handle combinatorial constraints
- Work with discrete probability distributions

Example 1:
Input: balls = [1,1]
Output: 1.0
Explanation: Only 2 balls, each of different color. Whatever way you put balls, the two boxes will have the same number of balls of each color.

Example 2:
Input: balls = [2,1,1]
Output: 0.66667
Explanation: We have 2 red, 1 blue, 1 yellow. The favorable outcomes are:
- Box 1: 1 red, 1 blue; Box 2: 1 red, 1 yellow
- Box 1: 1 red, 1 yellow; Box 2: 1 red, 1 blue
etc.

Constraints:
- 1 <= balls.length <= 8
- 1 <= balls[i] <= 6
- sum(balls) is even
"""


def probability_valid_sequence_recursive(balls):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion with combinatorial calculations.
    
    Time Complexity: O(n!) - exponential in ball combinations
    Space Complexity: O(n) - recursion stack
    """
    from math import factorial
    
    def comb(n, k):
        if k > n or k < 0:
            return 0
        return factorial(n) // (factorial(k) * factorial(n - k))
    
    total_balls = sum(balls)
    n_colors = len(balls)
    target_per_box = total_balls // 2
    
    def count_ways(color_idx, box1_count, box1_colors):
        if color_idx == n_colors:
            # Check if both boxes have equal counts and same total
            if box1_count == target_per_box and len(box1_colors) == n_colors:
                box2_colors = [balls[i] - box1_colors[i] for i in range(n_colors)]
                if all(c >= 0 for c in box2_colors) and box1_colors == box2_colors:
                    # Calculate multinomial coefficient
                    ways1 = factorial(target_per_box)
                    ways2 = factorial(target_per_box)
                    for i in range(n_colors):
                        ways1 //= factorial(box1_colors[i])
                        ways2 //= factorial(box2_colors[i])
                    return ways1 * ways2
            return 0
        
        total_ways = 0
        # Try putting 0 to balls[color_idx] of this color in box1
        for put_in_box1 in range(balls[color_idx] + 1):
            if box1_count + put_in_box1 <= target_per_box:
                new_box1_colors = box1_colors + [put_in_box1]
                total_ways += count_ways(color_idx + 1, box1_count + put_in_box1, new_box1_colors)
        
        return total_ways
    
    favorable = count_ways(0, 0, [])
    
    # Total ways to distribute balls
    total = factorial(total_balls)
    for ball_count in balls:
        total //= factorial(ball_count)
    total //= factorial(target_per_box) * factorial(target_per_box)
    
    return favorable / total if total > 0 else 0.0


def probability_valid_sequence_dp(balls):
    """
    DP APPROACH WITH MEMOIZATION:
    ============================
    Use DP to avoid recomputation.
    
    Time Complexity: O(n^k) - where k is number of colors
    Space Complexity: O(n^k) - memoization table
    """
    from math import factorial
    from functools import lru_cache
    
    total_balls = sum(balls)
    target_per_box = total_balls // 2
    n_colors = len(balls)
    
    @lru_cache(maxsize=None)
    def dp(color_idx, box1_count, box1_tuple):
        if color_idx == n_colors:
            if box1_count == target_per_box:
                box1_colors = list(box1_tuple)
                box2_colors = [balls[i] - box1_colors[i] for i in range(n_colors)]
                
                if all(c >= 0 for c in box2_colors) and box1_colors == box2_colors:
                    # Calculate multinomial coefficient
                    ways1 = factorial(target_per_box)
                    ways2 = factorial(target_per_box)
                    for i in range(n_colors):
                        ways1 //= factorial(box1_colors[i])
                        ways2 //= factorial(box2_colors[i])
                    return ways1 * ways2
            return 0
        
        total_ways = 0
        for put_in_box1 in range(balls[color_idx] + 1):
            if box1_count + put_in_box1 <= target_per_box:
                new_box1_tuple = box1_tuple + (put_in_box1,)
                total_ways += dp(color_idx + 1, box1_count + put_in_box1, new_box1_tuple)
        
        return total_ways
    
    favorable = dp(0, 0, ())
    
    # Total ways
    total = factorial(total_balls)
    for ball_count in balls:
        total //= factorial(ball_count)
    
    return favorable / total if total > 0 else 0.0


def probability_valid_sequence_optimized(balls):
    """
    OPTIMIZED COMBINATORIAL APPROACH:
    ================================
    Direct combinatorial calculation with pruning.
    
    Time Complexity: O(2^n) - optimized enumeration
    Space Complexity: O(1) - constant space
    """
    from math import factorial
    
    def multinomial(n, groups):
        result = factorial(n)
        for group in groups:
            result //= factorial(group)
        return result
    
    total_balls = sum(balls)
    if total_balls % 2 != 0:
        return 0.0
    
    target = total_balls // 2
    n_colors = len(balls)
    
    def backtrack(color_idx, box1_balls, box1_counts):
        if color_idx == n_colors:
            if box1_balls == target:
                # Check if remaining balls form valid second box
                box2_counts = [balls[i] - box1_counts[i] for i in range(n_colors)]
                if box1_counts == box2_counts:
                    return multinomial(target, box1_counts) * multinomial(target, box2_counts)
            return 0
        
        total_ways = 0
        max_take = min(balls[color_idx], target - box1_balls)
        
        for take in range(max_take + 1):
            total_ways += backtrack(color_idx + 1, box1_balls + take, box1_counts + [take])
        
        return total_ways
    
    favorable = backtrack(0, 0, [])
    total = multinomial(total_balls, balls)
    
    return favorable / total if total > 0 else 0.0


def probability_sequence_patterns():
    """
    SEQUENCE PATTERN ANALYSIS:
    =========================
    Analyze common probability sequence patterns.
    """
    
    def uniform_distribution_probability(n, target):
        """Probability in uniform distribution"""
        from math import comb
        return comb(n, target) / (2 ** n)
    
    def geometric_sequence_probability(p, k):
        """Probability of first success on k-th trial"""
        return (1 - p) ** (k - 1) * p
    
    def negative_binomial_probability(r, k, p):
        """Probability of k-th failure on r-th trial"""
        from math import comb
        return comb(k - 1, r - 1) * (p ** r) * ((1 - p) ** (k - r))
    
    def hypergeometric_probability(N, K, n, k):
        """Probability of k successes in n draws without replacement"""
        from math import comb
        return comb(K, k) * comb(N - K, n - k) / comb(N, n)
    
    def poisson_probability(lam, k):
        """Poisson probability"""
        from math import exp, factorial
        return (lam ** k) * exp(-lam) / factorial(k)
    
    # Example calculations
    patterns = {
        'uniform': uniform_distribution_probability(10, 5),
        'geometric': geometric_sequence_probability(0.3, 3),
        'negative_binomial': negative_binomial_probability(2, 5, 0.4),
        'hypergeometric': hypergeometric_probability(20, 7, 5, 2),
        'poisson': poisson_probability(3.5, 2)
    }
    
    return patterns


def probability_sequence_analysis(balls):
    """
    COMPREHENSIVE SEQUENCE ANALYSIS:
    ===============================
    Analyze probability sequences with detailed insights.
    """
    print(f"Probability Sequence Analysis:")
    print(f"Ball distribution: {balls}")
    print(f"Total balls: {sum(balls)}")
    print(f"Number of colors: {len(balls)}")
    
    if sum(balls) % 2 != 0:
        print("Invalid: Total balls must be even")
        return 0.0
    
    target_per_box = sum(balls) // 2
    print(f"Target per box: {target_per_box}")
    
    # Different approaches
    try:
        recursive = probability_valid_sequence_recursive(balls)
        print(f"Recursive result: {recursive:.6f}")
    except:
        print("Recursive: Too complex")
    
    try:
        dp_result = probability_valid_sequence_dp(balls)
        print(f"DP result: {dp_result:.6f}")
    except:
        print("DP: Memory limit exceeded")
    
    optimized = probability_valid_sequence_optimized(balls)
    print(f"Optimized result: {optimized:.6f}")
    
    # Analyze ball distribution
    print(f"\nBall Distribution Analysis:")
    total_arrangements = 1
    from math import factorial
    
    for i, count in enumerate(balls):
        print(f"  Color {i}: {count} balls")
    
    # Calculate total possible arrangements
    total_balls = sum(balls)
    total_arrangements = factorial(total_balls)
    for count in balls:
        total_arrangements //= factorial(count)
    
    print(f"Total arrangements: {total_arrangements}")
    
    # Combinatorial insights
    print(f"\nCombinatorial Insights:")
    print(f"• Equal distribution probability: {optimized:.4f}")
    print(f"• Favorable arrangements: {int(optimized * total_arrangements)}")
    print(f"• Unfavorable arrangements: {total_arrangements - int(optimized * total_arrangements)}")
    
    # Pattern analysis
    patterns = probability_sequence_patterns()
    print(f"\nCommon Probability Patterns:")
    for pattern, prob in patterns.items():
        print(f"  {pattern.capitalize()}: {prob:.4f}")
    
    return optimized


def probability_advanced_patterns():
    """
    ADVANCED PROBABILITY PATTERNS:
    =============================
    Demonstrate sophisticated probability calculations.
    """
    
    def martingale_probability(initial_value, target, max_steps):
        """Probability in martingale process"""
        # Simplified random walk with absorbing barriers
        if initial_value <= 0 or initial_value >= target:
            return 1.0 if initial_value >= target else 0.0
        
        # For simple symmetric random walk
        return initial_value / target
    
    def birth_death_probability(birth_rate, death_rate, initial, target, time_steps):
        """Birth-death process probability"""
        # Simplified calculation for demonstration
        if birth_rate == death_rate:
            return 1.0 / (time_steps + 1)  # Simplified
        
        ratio = death_rate / birth_rate
        if ratio != 1:
            numerator = 1 - ratio ** initial
            denominator = 1 - ratio ** target
            return numerator / denominator if denominator != 0 else 0.0
        
        return initial / target
    
    def branching_process_extinction(offspring_mean):
        """Extinction probability in branching process"""
        if offspring_mean <= 1:
            return 1.0
        else:
            # For Poisson offspring distribution
            # This requires solving s = exp(offspring_mean * (s - 1))
            # Simplified approximation
            return 1.0 / offspring_mean
    
    def queue_probability(arrival_rate, service_rate, max_customers):
        """Steady-state probability in M/M/1 queue"""
        if arrival_rate >= service_rate:
            return 1.0 / (max_customers + 1)  # Unstable queue
        
        rho = arrival_rate / service_rate
        if max_customers == 0:
            return 1 - rho
        
        return (1 - rho) * (rho ** max_customers)
    
    # Example calculations
    advanced_patterns = {
        'martingale': martingale_probability(5, 10, 100),
        'birth_death': birth_death_probability(0.3, 0.2, 3, 8, 20),
        'extinction': branching_process_extinction(1.5),
        'queue': queue_probability(0.8, 1.0, 3)
    }
    
    print("Advanced Probability Patterns:")
    print("=" * 40)
    
    for pattern, prob in advanced_patterns.items():
        print(f"{pattern.replace('_', ' ').title()}: {prob:.4f}")
    
    return advanced_patterns


# Test cases
def test_probability_sequences():
    """Test probability sequence implementations"""
    test_cases = [
        ([1, 1], 1.0),
        ([2, 1, 1], 0.66667),
        ([1, 2, 3], 0.0),  # Cannot be equally distributed
        ([2, 2], 0.5),
        ([3, 3], 0.4)
    ]
    
    print("Testing Probability Sequence Solutions:")
    print("=" * 70)
    
    for i, (balls, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"balls = {balls}")
        print(f"Expected: {expected}")
        
        if sum(balls) % 2 != 0:
            print("Invalid input: odd total")
            continue
        
        try:
            recursive = probability_valid_sequence_recursive(balls)
            diff = abs(recursive - expected)
            print(f"Recursive:        {recursive:.6f} {'✓' if diff < 0.001 else '✗'}")
        except:
            print("Recursive:        Too complex")
        
        try:
            dp_result = probability_valid_sequence_dp(balls)
            dp_diff = abs(dp_result - expected)
            print(f"DP:               {dp_result:.6f} {'✓' if dp_diff < 0.001 else '✗'}")
        except:
            print("DP:               Memory exceeded")
        
        optimized = probability_valid_sequence_optimized(balls)
        opt_diff = abs(optimized - expected)
        print(f"Optimized:        {optimized:.6f} {'✓' if opt_diff < 0.001 else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    probability_sequence_analysis([2, 1, 1])
    
    # Advanced patterns
    print(f"\n" + "=" * 70)
    probability_advanced_patterns()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. COMBINATORIAL COMPLEXITY: Exact counting requires careful enumeration")
    print("2. SYMMETRY CONDITIONS: Valid sequences require perfect balance")
    print("3. MULTINOMIAL CALCULATIONS: Multiple categories need multinomial coefficients")
    print("4. PRUNING OPTIMIZATION: Early termination reduces computational complexity")
    print("5. PATTERN RECOGNITION: Common probability distributions have closed forms")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Statistical Physics: Particle distribution in energy states")
    print("• Queueing Theory: Service distribution analysis")
    print("• Genetics: Allele frequency calculations")
    print("• Machine Learning: Balanced dataset probability")
    print("• Combinatorial Optimization: Fair allocation probability")


if __name__ == "__main__":
    test_probability_sequences()


"""
PROBABILITY OF VALID SEQUENCE - COMBINATORIAL PROBABILITY MASTERY:
==================================================================

This problem demonstrates advanced combinatorial probability:
- Multinomial distributions with balance constraints
- Exact enumeration using dynamic programming
- Combinatorial coefficient calculations
- Complex constraint satisfaction probability

KEY INSIGHTS:
============
1. **MULTINOMIAL STRUCTURE**: Multiple categories require multinomial coefficients
2. **BALANCE CONSTRAINTS**: Valid sequences must satisfy symmetry conditions
3. **COMBINATORIAL ENUMERATION**: Exact counting through systematic exploration
4. **OPTIMIZATION PRUNING**: Early termination when constraints violated
5. **MATHEMATICAL PRECISION**: Requires careful handling of large factorials

ALGORITHM APPROACHES:
====================

1. **Recursive Enumeration**: O(k^n) time, O(n) space
   - Direct enumeration of all valid distributions
   - Exponential complexity in number of arrangements

2. **DP with Memoization**: O(k^n) time, O(k^n) space
   - Cache intermediate results to avoid recomputation
   - State explosion for large inputs

3. **Optimized Backtracking**: O(2^k) time, O(k) space
   - Pruned search with early termination
   - Most practical for moderate-sized problems

COMBINATORIAL MATHEMATICS:
=========================
**Multinomial Coefficient**: C(n; k₁,k₂,...,kₘ) = n! / (k₁! × k₂! × ... × kₘ!)

**Total Arrangements**: Product of multinomial coefficients for each box
**Favorable Outcomes**: Arrangements satisfying balance constraints
**Probability**: Favorable / Total

CONSTRAINT ANALYSIS:
===================
**Balance Requirement**: Each color must have equal count in both boxes
**Feasibility**: Only possible when each color count is even
**Symmetry**: Problem has inherent symmetrical structure

OPTIMIZATION TECHNIQUES:
=======================
**Early Pruning**: Stop exploration when constraints cannot be satisfied
**Symmetry Exploitation**: Reduce search space using problem symmetries
**Memoization**: Cache results for repeated subproblems
**Mathematical Shortcuts**: Use closed forms when available

COMPLEXITY CONSIDERATIONS:
=========================
**State Space**: Exponential in number of colors and balls
**Factorial Growth**: Multinomial coefficients grow factorially
**Precision**: Large numbers require careful arithmetic
**Approximation**: May need approximation for very large instances

APPLICATIONS:
============
- **Statistical Physics**: Particle distribution in quantum states
- **Quality Control**: Balanced sampling probability
- **Genetics**: Allele frequency equilibrium
- **Game Theory**: Fair division probability
- **Machine Learning**: Balanced dataset generation

This problem showcases the intersection of combinatorics
and probability theory, requiring sophisticated mathematical
techniques for exact probability calculation in constrained
multinomial scenarios.
"""
