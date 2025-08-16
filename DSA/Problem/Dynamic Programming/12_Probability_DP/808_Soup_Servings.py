"""
LeetCode 808: Soup Servings
Difficulty: Medium
Category: Probability DP - Expected Value with Approximation

PROBLEM DESCRIPTION:
===================
There are two types of soup: type A and type B. Initially, we have n ml of each type of soup.

There are four kinds of operations:
1. Serve 100 ml of soup A and 0 ml of soup B
2. Serve 75 ml of soup A and 25 ml of soup B
3. Serve 50 ml of soup A and 50 ml of soup B
4. Serve 25 ml of soup A and 75 ml of soup B

Each operation has equal probability (25% each).

When one or both types of soup are served, the game ends. Return the probability that soup A will be empty first, plus half the probability that A and B become empty at the same time.

Note that we do not have an operation where all 100 ml's of soup B are used first.

Example 1:
Input: n = 50
Output: 0.625
Explanation: 
If we choose the first two operations, A will become empty first.
For the third operation, A and B will become empty at the same time.
For the fourth operation, B will become empty first.
So the total probability of A being empty first plus half the probability of A and B being empty at the same time, is 0.5 + 0.5 + 0.125 + 0.0 = 0.625.

Example 2:
Input: n = 100
Output: 0.71875

Constraints:
- 0 <= n <= 10^9
"""


def soup_servings_recursive(n):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to calculate probability directly.
    
    Time Complexity: O(4^(n/25)) - exponential without memoization
    Space Complexity: O(n/25) - recursion stack
    """
    # Operations: (soup A reduction, soup B reduction)
    operations = [(100, 0), (75, 25), (50, 50), (25, 75)]
    
    def probability(a, b):
        # Base cases
        if a <= 0 and b <= 0:
            return 0.5  # Both empty at same time
        if a <= 0:
            return 1.0  # A empty first
        if b <= 0:
            return 0.0  # B empty first
        
        total_prob = 0.0
        for da, db in operations:
            total_prob += probability(a - da, b - db)
        
        return total_prob / 4.0
    
    return probability(n, n)


def soup_servings_memoization(n):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed states.
    
    Time Complexity: O((n/25)^2) - quadratic in reduced space
    Space Complexity: O((n/25)^2) - memoization table
    """
    operations = [(100, 0), (75, 25), (50, 50), (25, 75)]
    memo = {}
    
    def probability(a, b):
        if a <= 0 and b <= 0:
            return 0.5
        if a <= 0:
            return 1.0
        if b <= 0:
            return 0.0
        
        if (a, b) in memo:
            return memo[(a, b)]
        
        total_prob = 0.0
        for da, db in operations:
            total_prob += probability(a - da, b - db)
        
        result = total_prob / 4.0
        memo[(a, b)] = result
        return result
    
    return probability(n, n)


def soup_servings_optimized(n):
    """
    OPTIMIZED APPROACH WITH SCALING:
    ===============================
    Scale down the problem and use approximation for large n.
    
    Time Complexity: O(min(n^2, 1)) - constant for large n
    Space Complexity: O(min(n^2, 1)) - constant for large n
    """
    # For large n, the probability approaches 1
    # This is because soup A is consumed faster on average
    if n >= 4800:  # Empirically determined threshold
        return 1.0
    
    # Scale down by 25 to reduce state space
    # Since all operations are multiples of 25
    n = (n + 24) // 25  # Ceiling division
    
    operations = [(4, 0), (3, 1), (2, 2), (1, 3)]  # Scaled operations
    memo = {}
    
    def probability(a, b):
        if a <= 0 and b <= 0:
            return 0.5
        if a <= 0:
            return 1.0
        if b <= 0:
            return 0.0
        
        if (a, b) in memo:
            return memo[(a, b)]
        
        total_prob = 0.0
        for da, db in operations:
            total_prob += probability(a - da, b - db)
        
        result = total_prob / 4.0
        memo[(a, b)] = result
        return result
    
    return probability(n, n)


def soup_servings_dp_bottom_up(n):
    """
    BOTTOM-UP DP APPROACH:
    =====================
    Use iterative DP with optimized scaling.
    
    Time Complexity: O(n^2) - for scaled n
    Space Complexity: O(n^2) - DP table
    """
    if n >= 4800:
        return 1.0
    
    # Scale down
    n = (n + 24) // 25
    
    # dp[i][j] = probability starting with i units of A and j units of B
    dp = {}
    
    # Base cases
    for i in range(n + 5):
        for j in range(n + 5):
            if i <= 0 and j <= 0:
                dp[(i, j)] = 0.5
            elif i <= 0:
                dp[(i, j)] = 1.0
            elif j <= 0:
                dp[(i, j)] = 0.0
    
    # Fill DP table
    operations = [(4, 0), (3, 1), (2, 2), (1, 3)]
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            total_prob = 0.0
            for da, db in operations:
                next_a = i - da
                next_b = j - db
                total_prob += dp.get((next_a, next_b), 0.0)
            
            dp[(i, j)] = total_prob / 4.0
    
    return dp.get((n, n), 0.0)


def soup_servings_with_analysis(n):
    """
    SOUP SERVINGS WITH DETAILED ANALYSIS:
    ====================================
    Solve the problem and provide comprehensive insights.
    
    Time Complexity: O(min(n^2, 1)) - optimized complexity
    Space Complexity: O(min(n^2, 1)) - DP table + analysis
    """
    original_n = n
    
    analysis = {
        'original_n': original_n,
        'scaled_n': None,
        'operations': [(100, 0), (75, 25), (50, 50), (25, 75)],
        'operation_analysis': {},
        'convergence_info': {},
        'probability_breakdown': {},
        'insights': []
    }
    
    # Analyze operations
    ops = analysis['operations']
    total_a_consumption = sum(op[0] for op in ops) / len(ops)
    total_b_consumption = sum(op[1] for op in ops) / len(ops)
    
    analysis['operation_analysis'] = {
        'avg_a_consumption': total_a_consumption,
        'avg_b_consumption': total_b_consumption,
        'a_bias': total_a_consumption - total_b_consumption,
        'operations_detail': [
            {'a_consumed': op[0], 'b_consumed': op[1], 'bias': op[0] - op[1]}
            for op in ops
        ]
    }
    
    # Large n approximation analysis
    if n >= 4800:
        analysis['convergence_info'] = {
            'uses_approximation': True,
            'threshold': 4800,
            'approximation_value': 1.0,
            'reason': 'Soup A consumed faster on average, probability approaches 1'
        }
        analysis['insights'].append(f"Large n ({n}) uses approximation: probability ≈ 1.0")
        return 1.0, analysis
    
    # Scale down for computation
    scaled_n = (n + 24) // 25
    analysis['scaled_n'] = scaled_n
    analysis['convergence_info'] = {
        'uses_approximation': False,
        'scaling_factor': 25,
        'scaled_n': scaled_n
    }
    
    # Compute with detailed tracking
    operations = [(4, 0), (3, 1), (2, 2), (1, 3)]
    memo = {}
    
    # Track some intermediate probabilities
    intermediate_probs = {}
    
    def probability(a, b):
        if a <= 0 and b <= 0:
            return 0.5
        if a <= 0:
            return 1.0
        if b <= 0:
            return 0.0
        
        if (a, b) in memo:
            return memo[(a, b)]
        
        total_prob = 0.0
        operation_contributions = []
        
        for i, (da, db) in enumerate(operations):
            next_prob = probability(a - da, b - db)
            operation_contributions.append(next_prob)
            total_prob += next_prob
        
        result = total_prob / 4.0
        memo[(a, b)] = result
        
        # Store some intermediate results for analysis
        if a == b and a <= 5:  # Small symmetric cases
            intermediate_probs[(a, b)] = {
                'probability': result,
                'operation_contributions': operation_contributions
            }
        
        return result
    
    final_probability = probability(scaled_n, scaled_n)
    
    analysis['probability_breakdown'] = {
        'final_probability': final_probability,
        'intermediate_cases': intermediate_probs
    }
    
    # Generate insights
    analysis['insights'].append(f"Final probability: {final_probability:.6f}")
    analysis['insights'].append(f"Soup A bias: {analysis['operation_analysis']['a_bias']:.1f} ml per operation")
    
    if final_probability > 0.5:
        analysis['insights'].append("Soup A more likely to empty first due to operation bias")
    
    # Analyze operation bias
    biased_ops = sum(1 for op in analysis['operation_analysis']['operations_detail'] if op['bias'] > 0)
    analysis['insights'].append(f"{biased_ops}/4 operations favor soup A consumption")
    
    # Expected game length approximation
    if scaled_n <= 10:
        expected_length = scaled_n / (total_a_consumption / 25)  # Very rough approximation
        analysis['insights'].append(f"Expected game length: ~{expected_length:.1f} operations")
    
    return final_probability, analysis


def soup_servings_analysis(n):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze soup servings with detailed insights.
    """
    print(f"Soup Servings Analysis:")
    print(f"Initial soup amount: {n} ml each")
    
    # Analyze operations
    operations = [(100, 0), (75, 25), (50, 50), (25, 75)]
    print(f"Operations (A reduction, B reduction):")
    for i, (a, b) in enumerate(operations, 1):
        bias = a - b
        print(f"  Op {i}: ({a:3d}, {b:2d}) - bias toward A: {bias:+3d}")
    
    avg_a = sum(op[0] for op in operations) / 4
    avg_b = sum(op[1] for op in operations) / 4
    print(f"Average per operation: A = {avg_a:.1f}, B = {avg_b:.1f}")
    print(f"Bias toward consuming A: {avg_a - avg_b:.1f} ml per operation")
    
    # Different approaches
    if n <= 200:
        try:
            recursive = soup_servings_recursive(n)
            print(f"Recursive result: {recursive:.6f}")
        except:
            print("Recursive: Too slow")
    
    if n <= 2000:
        memoization = soup_servings_memoization(n)
        print(f"Memoization result: {memoization:.6f}")
    
    optimized = soup_servings_optimized(n)
    print(f"Optimized result: {optimized:.6f}")
    
    # Detailed analysis
    detailed_result, analysis = soup_servings_with_analysis(n)
    
    print(f"\nDetailed Analysis:")
    print(f"Final probability: {detailed_result:.6f}")
    
    if analysis['convergence_info']['uses_approximation']:
        conv_info = analysis['convergence_info']
        print(f"Uses approximation: {conv_info['reason']}")
        print(f"Threshold: {conv_info['threshold']}")
    else:
        print(f"Scaling factor: {analysis['convergence_info']['scaling_factor']}")
        print(f"Scaled problem size: {analysis['convergence_info']['scaled_n']}")
    
    print(f"\nOperation Analysis:")
    op_analysis = analysis['operation_analysis']
    print(f"Average A consumption: {op_analysis['avg_a_consumption']:.1f} ml")
    print(f"Average B consumption: {op_analysis['avg_b_consumption']:.1f} ml")
    print(f"A consumption bias: {op_analysis['a_bias']:.1f} ml")
    
    print(f"\nInsights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    # Show intermediate probabilities for small cases
    if 'intermediate_cases' in analysis['probability_breakdown']:
        intermediate = analysis['probability_breakdown']['intermediate_cases']
        if intermediate:
            print(f"\nIntermediate Probabilities:")
            for (a, b), data in intermediate.items():
                print(f"  P(A wins | A={a}, B={b}) = {data['probability']:.4f}")
    
    return detailed_result


def soup_servings_variants():
    """
    SOUP SERVINGS VARIANTS:
    ======================
    Different operation sets and modifications.
    """
    
    def soup_servings_custom_ops(n, operations):
        """Soup servings with custom operations"""
        if n >= 4800:  # Still use approximation for large n
            return 1.0
        
        n = (n + 24) // 25
        memo = {}
        
        def probability(a, b):
            if a <= 0 and b <= 0:
                return 0.5
            if a <= 0:
                return 1.0
            if b <= 0:
                return 0.0
            
            if (a, b) in memo:
                return memo[(a, b)]
            
            total_prob = 0.0
            for da, db in operations:
                total_prob += probability(a - da, b - db)
            
            result = total_prob / len(operations)
            memo[(a, b)] = result
            return result
        
        return probability(n, n)
    
    def soup_servings_weighted_ops(n, operation_weights):
        """Soup servings with weighted operation probabilities"""
        operations = [(4, 0), (3, 1), (2, 2), (1, 3)]
        
        if n >= 4800:
            return 1.0
        
        n = (n + 24) // 25
        memo = {}
        
        def probability(a, b):
            if a <= 0 and b <= 0:
                return 0.5
            if a <= 0:
                return 1.0
            if b <= 0:
                return 0.0
            
            if (a, b) in memo:
                return memo[(a, b)]
            
            total_prob = 0.0
            for i, (da, db) in enumerate(operations):
                weight = operation_weights[i]
                total_prob += weight * probability(a - da, b - db)
            
            memo[(a, b)] = total_prob
            return total_prob
        
        return probability(n, n)
    
    def soup_servings_three_types(n):
        """Extended to three soup types"""
        # Simplified version - operations affect A, B, C
        operations = [
            (4, 0, 0), (3, 1, 0), (2, 2, 0), (1, 3, 0),
            (2, 1, 1), (1, 2, 1), (1, 1, 2), (0, 2, 2)
        ]
        
        if n >= 2000:  # Smaller threshold for 3D problem
            return 1.0 / 3  # Rough approximation
        
        n = (n + 24) // 25
        memo = {}
        
        def probability(a, b, c):
            # Count how many are empty
            empty_count = (a <= 0) + (b <= 0) + (c <= 0)
            
            if empty_count >= 2:
                return 0.5  # Multiple empty - tie scenario
            elif a <= 0:
                return 1.0  # A empty first
            elif empty_count == 1:
                return 0.0  # Other empty first
            
            if (a, b, c) in memo:
                return memo[(a, b, c)]
            
            total_prob = 0.0
            for da, db, dc in operations:
                total_prob += probability(a - da, b - db, c - dc)
            
            result = total_prob / len(operations)
            memo[(a, b, c)] = result
            return result
        
        return probability(n, n, n)
    
    # Test variants
    test_values = [50, 100, 500, 1000]
    
    print("Soup Servings Variants:")
    print("=" * 50)
    
    for n in test_values:
        print(f"\nn = {n}:")
        
        basic_result = soup_servings_optimized(n)
        print(f"Basic soup servings: {basic_result:.6f}")
        
        # Custom operations (more balanced)
        balanced_ops = [(3, 1), (2, 2), (2, 2), (1, 3)]  # Less A bias
        custom_result = soup_servings_custom_ops(n, balanced_ops)
        print(f"Balanced operations: {custom_result:.6f}")
        
        # Weighted operations (favor first operation)
        weights = [0.4, 0.2, 0.2, 0.2]  # First operation more likely
        weighted_result = soup_servings_weighted_ops(n, weights)
        print(f"Weighted operations: {weighted_result:.6f}")
        
        # Three soup types
        if n <= 500:  # Only for smaller n due to complexity
            three_type_result = soup_servings_three_types(n)
            print(f"Three soup types: {three_type_result:.6f}")


# Test cases
def test_soup_servings():
    """Test all implementations with various inputs"""
    test_cases = [
        (50, 0.625),
        (100, 0.71875),
        (0, 0.5),
        (25, 0.5625),
        (5000, 1.0)  # Large n should use approximation
    ]
    
    print("Testing Soup Servings Solutions:")
    print("=" * 70)
    
    for i, (n, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}")
        print(f"Expected: {expected}")
        
        # Skip recursive for large cases
        if n <= 150:
            try:
                recursive = soup_servings_recursive(n)
                diff = abs(recursive - expected)
                print(f"Recursive:        {recursive:.6f} {'✓' if diff < 0.001 else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        if n <= 2000:
            memoization = soup_servings_memoization(n)
            memo_diff = abs(memoization - expected)
            print(f"Memoization:      {memoization:.6f} {'✓' if memo_diff < 0.001 else '✗'}")
        
        optimized = soup_servings_optimized(n)
        opt_diff = abs(optimized - expected)
        print(f"Optimized:        {optimized:.6f} {'✓' if opt_diff < 0.001 else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    soup_servings_analysis(100)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    soup_servings_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. OPERATION BIAS: Soup A consumed faster on average (62.5 vs 37.5 ml)")
    print("2. CONVERGENCE: For large n, probability approaches 1 due to bias")
    print("3. SCALING OPTIMIZATION: Reduce state space by factor of 25")
    print("4. APPROXIMATION: Use threshold to avoid expensive computation")
    print("5. SYMMETRIC BREAKING: Tie condition gets 50% probability weight")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Resource Depletion: Modeling competing resource consumption")
    print("• Game Balance: Analyzing fairness in randomized game mechanics")
    print("• Reliability Analysis: System failure probability with multiple modes")
    print("• Queueing Theory: Service completion probability analysis")
    print("• Optimization: Expected value calculation in stochastic processes")


if __name__ == "__main__":
    test_soup_servings()


"""
SOUP SERVINGS - PROBABILITY WITH ASYMMETRIC BIAS AND APPROXIMATION:
===================================================================

This problem demonstrates probability DP with systematic bias:
- Asymmetric operations that favor one outcome over another
- Large state space requiring scaling and approximation techniques
- Convergence analysis for asymptotic behavior
- Expected value calculation with absorbing states

KEY INSIGHTS:
============
1. **OPERATION BIAS**: Soup A consumed faster on average (62.5 vs 37.5 ml per operation)
2. **ASYMPTOTIC CONVERGENCE**: For large n, probability approaches 1 due to systematic bias
3. **SCALING OPTIMIZATION**: Reduce state space by greatest common divisor (25)
4. **APPROXIMATION THRESHOLD**: Use empirical threshold to avoid expensive computation
5. **TIE-BREAKING**: Simultaneous emptying gets 50% probability weight

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(4^(n/25)) time, O(n/25) space
   - Direct probability tree exploration
   - Exponential complexity without memoization

2. **Memoization**: O((n/25)²) time, O((n/25)²) space
   - Top-down DP with state caching
   - Quadratic in scaled problem size

3. **Approximation**: O(1) time, O(1) space
   - Use threshold-based approximation for large n
   - Leverages convergence analysis

4. **Scaled DP**: O((n/25)²) time, O((n/25)²) space
   - Bottom-up DP with optimized state space
   - Most practical approach

CORE SCALED DP ALGORITHM:
========================
```python
def soupServings(n):
    if n >= 4800:  # Empirical convergence threshold
        return 1.0
    
    n = (n + 24) // 25  # Scale by GCD of operations
    operations = [(4,0), (3,1), (2,2), (1,3)]  # Scaled operations
    
    memo = {}
    
    def prob(a, b):
        if a <= 0 and b <= 0: return 0.5  # Tie
        if a <= 0: return 1.0             # A wins
        if b <= 0: return 0.0             # B wins
        
        if (a,b) in memo: return memo[(a,b)]
        
        total = sum(prob(a-da, b-db) for da,db in operations)
        memo[(a,b)] = total / 4.0
        return memo[(a,b)]
    
    return prob(n, n)
```

BIAS ANALYSIS:
=============
**Operations**: (100,0), (75,25), (50,50), (25,75)
**Average Consumption**: A = 62.5 ml, B = 37.5 ml per operation
**Systematic Bias**: 25 ml per operation favoring A depletion

**Convergence**: As n → ∞, P(A empty first) → 1

SCALING OPTIMIZATION:
====================
**Observation**: All operations are multiples of 25
**Scaling Factor**: Divide all quantities by 25
**State Reduction**: From O(n²) to O((n/25)²) states
**Precision**: No loss due to integer scaling

APPROXIMATION TECHNIQUE:
=======================
**Empirical Threshold**: n ≥ 4800 implies P ≈ 1.0
**Convergence Rate**: Exponential approach to asymptotic value
**Error Bound**: Negligible for practical purposes
**Computational Savings**: Constant time for large inputs

ABSORBING STATE ANALYSIS:
========================
**Terminal States**: 
- (a ≤ 0, b ≤ 0): Both empty → probability 0.5
- (a ≤ 0, b > 0): A empty first → probability 1.0  
- (a > 0, b ≤ 0): B empty first → probability 0.0

**Transition Probabilities**: Uniform 1/4 for each operation

COMPLEXITY ANALYSIS:
===================
**Unscaled**: O(n²) states, prohibitive for large n
**Scaled**: O((n/25)²) = O(n²/625) states, manageable
**Approximation**: O(1) for n ≥ threshold

**Memory**: Same as time complexity for memoization

CONVERGENCE PROPERTIES:
======================
**Bias Direction**: Systematic toward A depletion
**Rate**: Exponential convergence to P = 1
**Threshold**: Empirically n ≥ 4800 gives P > 0.99999
**Stability**: Robust across different scaling approaches

APPLICATIONS:
============
- **Resource Competition**: Modeling asymmetric resource depletion
- **Reliability Engineering**: Component failure analysis with bias
- **Game Design**: Balancing mechanics with inherent advantages
- **Queuing Systems**: Service completion with preferential processing
- **Financial Modeling**: Default probability with systematic risk

RELATED PROBLEMS:
================
- **Gambler's Ruin**: Biased random walk with absorbing barriers
- **Asymmetric Random Walk**: General class of biased stochastic processes
- **Competing Processes**: Multiple processes with different rates
- **Threshold Models**: Problems requiring approximation for large parameters

VARIANTS:
========
- **Different Operations**: Modify consumption patterns
- **Weighted Probabilities**: Non-uniform operation selection
- **Multiple Resources**: Extend to 3+ competing resources
- **Dynamic Operations**: Operations that change over time

EDGE CASES:
==========
- **n = 0**: Both empty initially → probability 0.5
- **Single Operation**: Would create deterministic bias
- **Balanced Operations**: Equal average consumption → probability 0.5
- **Extreme Bias**: One resource always consumed → probability approaches 1

OPTIMIZATION TECHNIQUES:
=======================
**Scaling**: Reduce state space by GCD
**Approximation**: Empirical thresholds for convergence
**Memoization**: Cache intermediate results
**Early Termination**: Stop when convergence achieved

This problem elegantly demonstrates how systematic bias
in stochastic processes leads to predictable asymptotic
behavior, showcasing practical approximation techniques
for managing computational complexity in probability DP.
"""
