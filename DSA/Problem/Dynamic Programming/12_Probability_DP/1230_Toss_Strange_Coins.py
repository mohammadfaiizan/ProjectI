"""
LeetCode 1230: Toss Strange Coins
Difficulty: Medium
Category: Probability DP - Conditional Probability with Constraints

PROBLEM DESCRIPTION:
===================
You have some coins. The i-th coin has a probability prob[i] of facing heads when tossed.

Return the probability of getting exactly target heads after tossing all coins.

Example 1:
Input: prob = [0.4], target = 1
Output: 0.4
Explanation: Only one coin, probability of getting 1 head is 0.4

Example 2:
Input: prob = [0.5, 0.5], target = 0
Output: 0.25
Explanation: Probability of getting 0 heads = P(TT) = 0.5 * 0.5 = 0.25

Example 3:
Input: prob = [0.5, 0.5], target = 1
Output: 0.5
Explanation: P(HT) + P(TH) = 0.5*0.5 + 0.5*0.5 = 0.5

Example 4:
Input: prob = [0.5, 0.5], target = 2
Output: 0.25

Constraints:
- 1 <= prob.length <= 1000
- 0 <= prob[i] <= 1
- 0 <= target <= prob.length
"""


def toss_strange_coins_recursive(prob, target):
    """
    RECURSIVE APPROACH:
    ==================
    Use recursion to calculate probability directly.
    
    Time Complexity: O(2^n) - exponential without memoization
    Space Complexity: O(n) - recursion stack
    """
    n = len(prob)
    
    def probability(index, heads_needed):
        # Base cases
        if heads_needed < 0:
            return 0.0  # Impossible - too many heads already
        
        if index == n:
            return 1.0 if heads_needed == 0 else 0.0
        
        if heads_needed > n - index:
            return 0.0  # Impossible - not enough coins left
        
        # Two choices: heads or tails
        prob_heads = prob[index] * probability(index + 1, heads_needed - 1)
        prob_tails = (1 - prob[index]) * probability(index + 1, heads_needed)
        
        return prob_heads + prob_tails
    
    return probability(0, target)


def toss_strange_coins_memoization(prob, target):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to cache computed states.
    
    Time Complexity: O(n * target) - each state computed once
    Space Complexity: O(n * target) - memoization table
    """
    n = len(prob)
    memo = {}
    
    def probability(index, heads_needed):
        if heads_needed < 0:
            return 0.0
        
        if index == n:
            return 1.0 if heads_needed == 0 else 0.0
        
        if heads_needed > n - index:
            return 0.0
        
        if (index, heads_needed) in memo:
            return memo[(index, heads_needed)]
        
        prob_heads = prob[index] * probability(index + 1, heads_needed - 1)
        prob_tails = (1 - prob[index]) * probability(index + 1, heads_needed)
        
        result = prob_heads + prob_tails
        memo[(index, heads_needed)] = result
        return result
    
    return probability(0, target)


def toss_strange_coins_dp_bottom_up(prob, target):
    """
    BOTTOM-UP DP APPROACH:
    =====================
    Use iterative DP building from base cases.
    
    Time Complexity: O(n * target) - nested loops
    Space Complexity: O(target) - DP array
    """
    n = len(prob)
    
    if target > n:
        return 0.0
    
    # dp[j] = probability of getting exactly j heads so far
    dp = [0.0] * (target + 1)
    dp[0] = 1.0  # Base case: 0 heads with 0 coins
    
    for i in range(n):
        # Process in reverse order to avoid overwriting
        new_dp = [0.0] * (target + 1)
        
        for j in range(target + 1):
            if dp[j] > 0:
                # Don't flip this coin (tails)
                new_dp[j] += dp[j] * (1 - prob[i])
                
                # Flip this coin (heads)
                if j + 1 <= target:
                    new_dp[j + 1] += dp[j] * prob[i]
        
        dp = new_dp
    
    return dp[target]


def toss_strange_coins_space_optimized(prob, target):
    """
    SPACE-OPTIMIZED DP:
    ==================
    Use single DP array with careful update order.
    
    Time Complexity: O(n * target) - same as bottom-up
    Space Complexity: O(target) - single DP array
    """
    n = len(prob)
    
    if target > n:
        return 0.0
    
    # dp[j] = probability of getting exactly j heads
    dp = [0.0] * (target + 1)
    dp[0] = 1.0
    
    for i in range(n):
        # Update in reverse order to avoid overwriting needed values
        for j in range(min(target, i + 1), -1, -1):
            if j > 0:
                # Getting j heads: either had j-1 and this coin is heads
                # or had j heads and this coin is tails
                dp[j] = dp[j] * (1 - prob[i]) + dp[j - 1] * prob[i]
            else:
                # Getting 0 heads: had 0 heads and this coin is tails
                dp[0] = dp[0] * (1 - prob[i])
    
    return dp[target]


def toss_strange_coins_with_analysis(prob, target):
    """
    TOSS STRANGE COINS WITH DETAILED ANALYSIS:
    =========================================
    Solve the problem and provide comprehensive insights.
    
    Time Complexity: O(n * target) - DP computation + analysis
    Space Complexity: O(n * target) - DP table + analysis data
    """
    n = len(prob)
    
    analysis = {
        'num_coins': n,
        'target_heads': target,
        'coin_probabilities': prob[:],
        'probability_distribution': {},
        'coin_analysis': {},
        'statistical_measures': {},
        'insights': []
    }
    
    if target > n:
        analysis['insights'].append(f"Impossible: target {target} > number of coins {n}")
        return 0.0, analysis
    
    # Analyze individual coins
    expected_heads = sum(prob)
    variance = sum(p * (1 - p) for p in prob)
    
    analysis['coin_analysis'] = {
        'expected_total_heads': expected_heads,
        'variance': variance,
        'standard_deviation': variance ** 0.5,
        'most_likely_outcome': round(expected_heads),
        'coin_details': [
            {
                'index': i,
                'probability': p,
                'expected_contribution': p,
                'variance_contribution': p * (1 - p)
            }
            for i, p in enumerate(prob)
        ]
    }
    
    # Calculate full probability distribution
    dp = [[0.0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1.0
    
    # Track the full distribution
    for i in range(n):
        for j in range(i + 2):  # j can be at most i+1
            if dp[i][j] > 0:
                # Tails
                dp[i + 1][j] += dp[i][j] * (1 - prob[i])
                # Heads
                if j + 1 <= n:
                    dp[i + 1][j + 1] += dp[i][j] * prob[i]
    
    # Store full distribution
    full_distribution = {}
    for j in range(n + 1):
        if dp[n][j] > 1e-10:  # Only store significant probabilities
            full_distribution[j] = dp[n][j]
    
    analysis['probability_distribution'] = full_distribution
    
    # Calculate statistical measures
    calculated_expected = sum(heads * prob for heads, prob in full_distribution.items())
    calculated_variance = sum((heads - calculated_expected) ** 2 * prob 
                            for heads, prob in full_distribution.items())
    
    analysis['statistical_measures'] = {
        'calculated_expected_heads': calculated_expected,
        'calculated_variance': calculated_variance,
        'mode': max(full_distribution.items(), key=lambda x: x[1])[0],
        'probability_at_target': dp[n][target]
    }
    
    # Generate insights
    result = dp[n][target]
    analysis['insights'].append(f"Probability of exactly {target} heads: {result:.6f}")
    analysis['insights'].append(f"Expected number of heads: {expected_heads:.2f}")
    
    # Compare target to expected value
    if target < expected_heads - 1:
        analysis['insights'].append(f"Target {target} is significantly below expected {expected_heads:.1f}")
    elif target > expected_heads + 1:
        analysis['insights'].append(f"Target {target} is significantly above expected {expected_heads:.1f}")
    else:
        analysis['insights'].append(f"Target {target} is close to expected {expected_heads:.1f}")
    
    # Analyze coin fairness
    fair_coins = sum(1 for p in prob if abs(p - 0.5) < 0.01)
    biased_coins = n - fair_coins
    
    if fair_coins == n:
        analysis['insights'].append("All coins are fair (p ≈ 0.5)")
    elif biased_coins == n:
        analysis['insights'].append("All coins are biased")
    else:
        analysis['insights'].append(f"{fair_coins} fair coins, {biased_coins} biased coins")
    
    # Most/least likely outcomes
    if len(full_distribution) > 1:
        max_prob_heads = max(full_distribution.items(), key=lambda x: x[1])
        min_prob_heads = min(full_distribution.items(), key=lambda x: x[1])
        analysis['insights'].append(f"Most likely: {max_prob_heads[0]} heads (p={max_prob_heads[1]:.4f})")
        analysis['insights'].append(f"Least likely: {min_prob_heads[0]} heads (p={min_prob_heads[1]:.4f})")
    
    return result, analysis


def toss_strange_coins_analysis(prob, target):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze strange coin tossing with detailed insights.
    """
    print(f"Strange Coin Tossing Analysis:")
    print(f"Number of coins: {len(prob)}")
    print(f"Target heads: {target}")
    print(f"Coin probabilities: {prob}")
    
    # Basic statistics
    expected = sum(prob)
    variance = sum(p * (1 - p) for p in prob)
    std_dev = variance ** 0.5
    
    print(f"Expected heads: {expected:.3f}")
    print(f"Standard deviation: {std_dev:.3f}")
    print(f"Variance: {variance:.3f}")
    
    # Different approaches
    if len(prob) <= 12:
        try:
            recursive = toss_strange_coins_recursive(prob, target)
            print(f"Recursive result: {recursive:.6f}")
        except:
            print("Recursive: Too slow")
    
    memoization = toss_strange_coins_memoization(prob, target)
    bottom_up = toss_strange_coins_dp_bottom_up(prob, target)
    optimized = toss_strange_coins_space_optimized(prob, target)
    
    print(f"Memoization result: {memoization:.6f}")
    print(f"Bottom-up DP result: {bottom_up:.6f}")
    print(f"Space optimized result: {optimized:.6f}")
    
    # Detailed analysis
    detailed_result, analysis = toss_strange_coins_with_analysis(prob, target)
    
    print(f"\nDetailed Analysis:")
    print(f"Target probability: {detailed_result:.6f}")
    
    print(f"\nCoin Analysis:")
    coin_analysis = analysis['coin_analysis']
    print(f"Expected total heads: {coin_analysis['expected_total_heads']:.3f}")
    print(f"Standard deviation: {coin_analysis['standard_deviation']:.3f}")
    print(f"Most likely outcome: {coin_analysis['most_likely_outcome']} heads")
    
    print(f"\nProbability Distribution:")
    distribution = analysis['probability_distribution']
    for heads in sorted(distribution.keys()):
        prob_val = distribution[heads]
        bar = "*" * min(int(prob_val * 50), 20)  # Visual bar
        print(f"  {heads:2d} heads: {prob_val:.4f} {bar}")
    
    print(f"\nInsights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    return detailed_result


def toss_strange_coins_variants():
    """
    STRANGE COIN VARIANTS:
    =====================
    Different scenarios and generalizations.
    """
    
    def at_least_k_heads(prob, k):
        """Probability of getting at least k heads"""
        n = len(prob)
        total_prob = 0.0
        
        for target in range(k, n + 1):
            total_prob += toss_strange_coins_space_optimized(prob, target)
        
        return total_prob
    
    def expected_heads_squared(prob):
        """Calculate E[X^2] where X is number of heads"""
        n = len(prob)
        
        # dp[i][j] = probability of j heads after i coins
        dp = [[0.0] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 1.0
        
        for i in range(n):
            for j in range(i + 2):
                if dp[i][j] > 0:
                    # Tails
                    dp[i + 1][j] += dp[i][j] * (1 - prob[i])
                    # Heads
                    if j + 1 <= n:
                        dp[i + 1][j + 1] += dp[i][j] * prob[i]
        
        expected_x_squared = sum(j * j * dp[n][j] for j in range(n + 1))
        return expected_x_squared
    
    def coin_importance_analysis(prob, target):
        """Analyze which coins are most important for reaching target"""
        n = len(prob)
        importance_scores = []
        
        for i in range(n):
            # Calculate probability with and without this coin contributing
            original_prob = prob[i]
            
            # Force coin to always be tails
            prob[i] = 0.0
            prob_without = toss_strange_coins_space_optimized(prob, target)
            
            # Force coin to always be heads
            prob[i] = 1.0
            prob_with = toss_strange_coins_space_optimized(prob, max(0, target - 1))
            
            # Restore original probability
            prob[i] = original_prob
            
            # Importance = how much this coin affects the target probability
            importance = abs(prob_with - prob_without)
            importance_scores.append((i, importance))
        
        return sorted(importance_scores, key=lambda x: x[1], reverse=True)
    
    def conditional_probability(prob, target, condition_heads, condition_indices):
        """P(total = target | specific coins show heads)"""
        # This is a simplified calculation
        n = len(prob)
        
        # Remove conditioned coins
        remaining_prob = [prob[i] for i in range(n) if i not in condition_indices]
        remaining_target = target - condition_heads
        
        if remaining_target < 0 or remaining_target > len(remaining_prob):
            return 0.0
        
        return toss_strange_coins_space_optimized(remaining_prob, remaining_target)
    
    # Test variants
    test_cases = [
        ([0.4], 1),
        ([0.5, 0.5], 1),
        ([0.3, 0.7, 0.5], 2),
        ([0.1, 0.9, 0.5, 0.5], 2)
    ]
    
    print("Strange Coin Tossing Variants:")
    print("=" * 50)
    
    for prob, target in test_cases:
        print(f"\nProb: {prob}, Target: {target}")
        
        basic_result = toss_strange_coins_space_optimized(prob, target)
        print(f"Exact target: {basic_result:.6f}")
        
        # At least k heads
        at_least_result = at_least_k_heads(prob, target)
        print(f"At least {target}: {at_least_result:.6f}")
        
        # Expected X^2
        if len(prob) <= 8:
            exp_x_squared = expected_heads_squared(prob)
            exp_x = sum(prob)
            variance = exp_x_squared - exp_x * exp_x
            print(f"E[X²]: {exp_x_squared:.3f}, Var(X): {variance:.3f}")
        
        # Coin importance
        if len(prob) <= 6:
            importance = coin_importance_analysis(prob, target)
            print(f"Most important coin: {importance[0][0]} (score: {importance[0][1]:.4f})")
        
        # Conditional probability
        if len(prob) >= 2:
            cond_prob = conditional_probability(prob, target, 1, [0])
            print(f"P(target | coin 0 = heads): {cond_prob:.6f}")


# Test cases
def test_toss_strange_coins():
    """Test all implementations with various inputs"""
    test_cases = [
        ([0.4], 1, 0.4),
        ([0.5, 0.5], 0, 0.25),
        ([0.5, 0.5], 1, 0.5),
        ([0.5, 0.5], 2, 0.25),
        ([0.3, 0.7, 0.5], 2, 0.44),
        ([1.0, 0.0], 1, 1.0)
    ]
    
    print("Testing Strange Coin Tossing Solutions:")
    print("=" * 70)
    
    for i, (prob, target, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"prob = {prob}, target = {target}")
        print(f"Expected: {expected}")
        
        # Skip recursive for large cases
        if len(prob) <= 10:
            try:
                recursive = toss_strange_coins_recursive(prob, target)
                diff = abs(recursive - expected)
                print(f"Recursive:        {recursive:.6f} {'✓' if diff < 0.001 else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
        memoization = toss_strange_coins_memoization(prob, target)
        bottom_up = toss_strange_coins_dp_bottom_up(prob, target)
        optimized = toss_strange_coins_space_optimized(prob, target)
        
        memo_diff = abs(memoization - expected)
        dp_diff = abs(bottom_up - expected)
        opt_diff = abs(optimized - expected)
        
        print(f"Memoization:      {memoization:.6f} {'✓' if memo_diff < 0.001 else '✗'}")
        print(f"Bottom-up DP:     {bottom_up:.6f} {'✓' if dp_diff < 0.001 else '✗'}")
        print(f"Space Optimized:  {optimized:.6f} {'✓' if opt_diff < 0.001 else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    toss_strange_coins_analysis([0.3, 0.7, 0.5], 2)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    toss_strange_coins_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. BINOMIAL GENERALIZATION: Extends binomial distribution to non-identical coins")
    print("2. CONDITIONAL PROBABILITY: Each coin contributes independently to final outcome")
    print("3. DYNAMIC PROGRAMMING: Optimal substructure allows efficient computation")
    print("4. STATISTICAL ANALYSIS: Full distribution reveals outcome probabilities")
    print("5. IMPORTANCE RANKING: Individual coin contributions can be quantified")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Risk Assessment: Probability of specific outcomes in uncertain systems")
    print("• Quality Control: Expected defect counts with varying failure rates")
    print("• Portfolio Analysis: Probability of achieving target returns")
    print("• Machine Learning: Feature importance in probabilistic models")
    print("• Reliability Engineering: System success probability calculation")


if __name__ == "__main__":
    test_toss_strange_coins()


"""
TOSS STRANGE COINS - GENERALIZED BINOMIAL DISTRIBUTION:
=======================================================

This problem demonstrates probability DP for generalized binomial scenarios:
- Non-identical coin probabilities requiring individual analysis
- Exact probability calculation for specific target outcomes
- Full probability distribution computation and analysis
- Statistical measure calculation (expectation, variance, mode)

KEY INSIGHTS:
============
1. **BINOMIAL GENERALIZATION**: Extends classical binomial to non-identical trials
2. **INDEPENDENCE**: Each coin flip is independent, enabling DP decomposition
3. **CONDITIONAL PROBABILITY**: Target achievement depends on exact head count
4. **STATISTICAL COMPLETENESS**: Can compute full outcome distribution
5. **OPTIMAL SUBSTRUCTURE**: Probability from position depends on remaining coins

ALGORITHM APPROACHES:
====================

1. **Recursive**: O(2^n) time, O(n) space
   - Direct enumeration of all possible outcomes
   - Exponential complexity without memoization

2. **Memoization**: O(n × target) time, O(n × target) space
   - Top-down DP with state caching
   - Each (position, heads_needed) computed once

3. **Bottom-up DP**: O(n × target) time, O(target) space
   - Iterative DP building probability distribution
   - Space-efficient with rolling array

4. **Space Optimized**: O(n × target) time, O(target) space
   - Single array with careful update ordering
   - Optimal space complexity

CORE DP ALGORITHM:
=================
```python
def tossStraangeCoins(prob, target):
    n = len(prob)
    if target > n: return 0.0
    
    # dp[j] = probability of exactly j heads so far
    dp = [0.0] * (target + 1)
    dp[0] = 1.0
    
    for i in range(n):
        # Update in reverse order to avoid overwriting
        for j in range(min(target, i + 1), -1, -1):
            if j > 0:
                dp[j] = dp[j] * (1 - prob[i]) + dp[j-1] * prob[i]
            else:
                dp[0] *= (1 - prob[i])
    
    return dp[target]
```

STATE TRANSITION ANALYSIS:
=========================
**State Definition**: dp[i][j] = probability of exactly j heads using first i coins
**Base Case**: dp[0][0] = 1.0 (zero heads with zero coins)
**Transition**: dp[i+1][j] = dp[i][j] × (1-p[i]) + dp[i][j-1] × p[i]

**Interpretation**: 
- dp[i][j] × (1-p[i]): j heads from i coins, (i+1)th coin is tails
- dp[i][j-1] × p[i]: (j-1) heads from i coins, (i+1)th coin is heads

STATISTICAL MEASURES:
====================
**Expected Value**: E[X] = Σᵢ p[i] (sum of individual probabilities)
**Variance**: Var(X) = Σᵢ p[i](1-p[i]) (sum of individual variances)
**Standard Deviation**: σ = √Var(X)

**Distribution**: Full probability mass function computable via DP

PROBABILITY DISTRIBUTION ANALYSIS:
=================================
**Complete Distribution**: P(X = k) for k = 0, 1, ..., n
**Mode**: Most likely number of heads
**Tail Probabilities**: P(X ≥ k), P(X ≤ k) for any threshold k
**Percentiles**: kth percentile of head count distribution

COMPLEXITY OPTIMIZATION:
=======================
**Space Reduction**: From O(n × target) to O(target) using rolling array
**Early Termination**: Stop if target becomes impossible
**Pruning**: Skip impossible states (heads_needed > coins_remaining)

**Update Order**: Reverse iteration prevents overwriting needed values

APPLICATIONS:
============
- **Risk Assessment**: Probability of achieving specific success counts
- **Quality Control**: Expected defect analysis with varying failure rates
- **Portfolio Management**: Probability of target returns with different assets
- **Reliability Engineering**: System success probability with component variations
- **A/B Testing**: Conversion probability analysis with heterogeneous groups

RELATED PROBLEMS:
================
- **Classical Binomial**: Special case where all p[i] are equal
- **Negative Binomial**: Number of trials until k successes
- **Hypergeometric**: Sampling without replacement scenarios
- **Poisson Binomial**: Theoretical foundation for this problem type

VARIANTS:
========
- **At Least K**: Probability of ≥ k heads (sum multiple exact probabilities)
- **Range Probability**: P(a ≤ X ≤ b) for head count ranges
- **Conditional Analysis**: P(target | subset of coins fixed)
- **Importance Ranking**: Which coins most affect target probability

EDGE CASES:
==========
- **target > n**: Impossible, probability = 0
- **target = 0**: All coins must be tails
- **All p[i] = 0**: Only 0 heads possible
- **All p[i] = 1**: Only n heads possible

NUMERICAL CONSIDERATIONS:
========================
**Precision**: Use double precision for accurate small probabilities
**Underflow**: Handle very small probabilities gracefully
**Normalization**: Verify probability distribution sums to 1.0
**Stability**: Maintain numerical stability across iterations

This problem beautifully generalizes the classical binomial
distribution to heterogeneous scenarios, demonstrating how
dynamic programming enables exact probability calculation
for complex stochastic processes while providing comprehensive
statistical analysis of outcome distributions.
"""
