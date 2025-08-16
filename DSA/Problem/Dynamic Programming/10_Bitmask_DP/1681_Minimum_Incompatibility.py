"""
LeetCode 1681: Minimum Incompatibility
Difficulty: Hard
Category: Bitmask DP - Subset Optimization with Constraints

PROBLEM DESCRIPTION:
===================
You have n integers and you want to partition them into k non-empty subsets. After the partitioning, you can get a score by computing the difference between the maximum and minimum value in each subset and then summing up all these differences.

For example, if you partition the array [1,2,1,4] into subsets [1,1] and [2,4], the score is (1-1) + (4-2) = 0 + 2 = 2.

Return the minimum possible score. If there is no valid partition (i.e., the partition must have exactly k non-empty subsets), return -1.

Example 1:
Input: nums = [1,2,1,4], k = 2
Output: 2
Explanation: Optimal partition is [1,1], [2,4]. Score = (1-1) + (4-2) = 0 + 2 = 2.

Example 2:
Input: nums = [6,3,8,1,3,1,2,2], k = 4
Output: 6
Explanation: Optimal partition is [1,1], [2,2], [3,3], [6,8]. Score = (1-1) + (2-2) + (3-3) + (8-6) = 0 + 0 + 0 + 2 = 2.

Example 3:
Input: nums = [5,3,3,6,3,3], k = 2
Output: -1
Explanation: It's impossible to partition into 2 subsets such that each subset has unique elements.

Constraints:
- 1 <= k <= nums.length <= 16
- nums.length is divisible by k
- 1 <= nums[i] <= nums.length
"""


def minimum_incompatibility_backtrack(nums, k):
    """
    BACKTRACKING APPROACH:
    =====================
    Try all possible partitions using backtracking.
    
    Time Complexity: O(k^n) - exponential
    Space Complexity: O(n) - recursion stack
    """
    from collections import Counter
    
    n = len(nums)
    if n % k != 0:
        return -1
    
    subset_size = n // k
    
    # Check if partition is possible
    count = Counter(nums)
    if any(freq > k for freq in count.values()):
        return -1  # Some number appears too often
    
    nums.sort()
    min_score = [float('inf')]
    
    def backtrack(idx, subsets, current_score):
        if idx == n:
            if len(subsets) == k and all(len(subset) == subset_size for subset in subsets):
                min_score[0] = min(min_score[0], current_score)
            return
        
        if current_score >= min_score[0]:
            return  # Pruning
        
        # Try adding current number to existing subsets
        for i, subset in enumerate(subsets):
            if len(subset) < subset_size and nums[idx] not in subset:
                new_subset = subset | {nums[idx]}
                if len(new_subset) == subset_size:
                    new_score = max(new_subset) - min(new_subset)
                    backtrack(idx + 1, subsets[:i] + [new_subset] + subsets[i+1:], 
                             current_score + new_score)
                else:
                    backtrack(idx + 1, subsets[:i] + [new_subset] + subsets[i+1:], 
                             current_score)
        
        # Try creating new subset
        if len(subsets) < k:
            backtrack(idx + 1, subsets + [{nums[idx]}], current_score)
    
    backtrack(0, [], 0)
    return min_score[0] if min_score[0] != float('inf') else -1


def minimum_incompatibility_bitmask_dp(nums, k):
    """
    BITMASK DP APPROACH:
    ===================
    Use bitmask DP to find optimal partition.
    
    Time Complexity: O(3^n) - iterate through all subset partitions
    Space Complexity: O(2^n) - DP table
    """
    from collections import Counter
    
    n = len(nums)
    if n % k != 0:
        return -1
    
    subset_size = n // k
    
    # Check feasibility
    count = Counter(nums)
    if any(freq > k for freq in count.values()):
        return -1
    
    # Precompute valid subsets and their scores
    valid_subsets = {}
    for mask in range(1, 1 << n):
        if bin(mask).count('1') != subset_size:
            continue
        
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        if len(set(subset)) == len(subset):  # All elements unique
            score = max(subset) - min(subset)
            valid_subsets[mask] = score
    
    # dp[mask] = minimum score to partition elements in mask
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    
    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue
        
        # Try all valid subsets that don't overlap with current mask
        for subset_mask, score in valid_subsets.items():
            if (mask & subset_mask) == 0:  # No overlap
                new_mask = mask | subset_mask
                dp[new_mask] = min(dp[new_mask], dp[mask] + score)
    
    full_mask = (1 << n) - 1
    return dp[full_mask] if dp[full_mask] != float('inf') else -1


def minimum_incompatibility_optimized_dp(nums, k):
    """
    OPTIMIZED BITMASK DP:
    ====================
    Use subset enumeration with pruning for better performance.
    
    Time Complexity: O(3^n) - optimal for this problem
    Space Complexity: O(2^n) - DP table
    """
    from collections import Counter
    
    n = len(nums)
    if n % k != 0:
        return -1
    
    subset_size = n // k
    count = Counter(nums)
    
    if any(freq > k for freq in count.values()):
        return -1
    
    # Generate all valid subsets
    valid_subsets = []
    for mask in range(1, 1 << n):
        if bin(mask).count('1') != subset_size:
            continue
        
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        if len(set(subset)) == subset_size:  # All unique
            score = max(subset) - min(subset)
            valid_subsets.append((mask, score))
    
    # Memoization for DP
    memo = {}
    
    def dp(mask, subsets_used):
        if subsets_used == k:
            return 0 if mask == (1 << n) - 1 else float('inf')
        
        if (mask, subsets_used) in memo:
            return memo[(mask, subsets_used)]
        
        result = float('inf')
        
        # Try each valid subset that doesn't conflict with current mask
        for subset_mask, score in valid_subsets:
            if (mask & subset_mask) == 0:  # No overlap
                new_mask = mask | subset_mask
                result = min(result, score + dp(new_mask, subsets_used + 1))
        
        memo[(mask, subsets_used)] = result
        return result
    
    result = dp(0, 0)
    return result if result != float('inf') else -1


def minimum_incompatibility_with_analysis(nums, k):
    """
    MINIMUM INCOMPATIBILITY WITH DETAILED ANALYSIS:
    ==============================================
    Solve the partition problem with detailed insights.
    
    Time Complexity: O(3^n) - standard approach
    Space Complexity: O(2^n) - DP table + analysis
    """
    from collections import Counter
    
    n = len(nums)
    subset_size = n // k
    
    analysis = {
        'n': n,
        'k': k,
        'subset_size': subset_size,
        'nums': nums[:],
        'feasible': True,
        'element_frequencies': {},
        'valid_subsets': [],
        'total_valid_subsets': 0,
        'optimal_partition': None,
        'optimal_score': float('inf')
    }
    
    # Basic feasibility checks
    if n % k != 0:
        analysis['feasible'] = False
        analysis['reason'] = f"n={n} not divisible by k={k}"
        return -1, analysis
    
    count = Counter(nums)
    analysis['element_frequencies'] = dict(count)
    
    if any(freq > k for freq in count.values()):
        analysis['feasible'] = False
        analysis['reason'] = f"Element appears more than k={k} times"
        return -1, analysis
    
    # Generate and analyze valid subsets
    valid_subsets = []
    for mask in range(1, 1 << n):
        if bin(mask).count('1') != subset_size:
            continue
        
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        if len(set(subset)) == subset_size:
            score = max(subset) - min(subset)
            valid_subsets.append((mask, subset, score))
    
    analysis['valid_subsets'] = [(subset, score) for _, subset, score in valid_subsets]
    analysis['total_valid_subsets'] = len(valid_subsets)
    
    if len(valid_subsets) == 0:
        analysis['feasible'] = False
        analysis['reason'] = "No valid subsets of required size with unique elements"
        return -1, analysis
    
    # DP with path tracking
    dp = [float('inf')] * (1 << n)
    parent = [None] * (1 << n)
    dp[0] = 0
    
    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue
        
        for subset_mask, subset, score in valid_subsets:
            if (mask & subset_mask) == 0:
                new_mask = mask | subset_mask
                new_score = dp[mask] + score
                
                if new_score < dp[new_mask]:
                    dp[new_mask] = new_score
                    parent[new_mask] = (mask, subset_mask, subset, score)
    
    full_mask = (1 << n) - 1
    optimal_score = dp[full_mask]
    
    if optimal_score == float('inf'):
        analysis['feasible'] = False
        analysis['reason'] = "No valid complete partition found"
        return -1, analysis
    
    # Reconstruct optimal partition
    partition = []
    current_mask = full_mask
    
    while current_mask != 0 and parent[current_mask]:
        prev_mask, subset_mask, subset, score = parent[current_mask]
        partition.append((subset, score))
        current_mask = prev_mask
    
    analysis['optimal_score'] = optimal_score
    analysis['optimal_partition'] = partition
    analysis['partition_scores'] = [score for _, score in partition]
    
    return optimal_score, analysis


def minimum_incompatibility_analysis(nums, k):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the minimum incompatibility problem with detailed insights.
    """
    print(f"Minimum Incompatibility Analysis:")
    print(f"Numbers: {nums}")
    print(f"Number of subsets k: {k}")
    print(f"Array length n: {len(nums)}")
    
    if len(nums) % k != 0:
        print(f"Impossible: n={len(nums)} not divisible by k={k}")
        return -1
    
    subset_size = len(nums) // k
    print(f"Required subset size: {subset_size}")
    
    from collections import Counter
    count = Counter(nums)
    print(f"Element frequencies: {dict(count)}")
    
    if any(freq > k for freq in count.values()):
        print(f"Impossible: Some element appears more than k={k} times")
        return -1
    
    # Different approaches
    if len(nums) <= 10:
        try:
            backtrack = minimum_incompatibility_backtrack(nums, k)
            print(f"Backtracking result: {backtrack}")
        except:
            print("Backtracking: Too slow")
    
    bitmask_dp = minimum_incompatibility_bitmask_dp(nums, k)
    optimized = minimum_incompatibility_optimized_dp(nums, k)
    
    print(f"Bitmask DP result: {bitmask_dp}")
    print(f"Optimized DP result: {optimized}")
    
    # Detailed analysis
    detailed_score, analysis = minimum_incompatibility_with_analysis(nums, k)
    
    print(f"\nDetailed Analysis:")
    print(f"Feasible: {analysis['feasible']}")
    
    if not analysis['feasible']:
        print(f"Reason: {analysis['reason']}")
        return -1
    
    print(f"Total valid subsets: {analysis['total_valid_subsets']}")
    print(f"Optimal score: {analysis['optimal_score']}")
    
    if analysis['optimal_partition']:
        print(f"\nOptimal Partition:")
        total_score = 0
        for i, (subset, score) in enumerate(analysis['optimal_partition']):
            print(f"  Subset {i+1}: {subset} (score: {score})")
            total_score += score
        print(f"  Total score: {total_score}")
    
    print(f"\nValid Subsets Analysis:")
    if len(analysis['valid_subsets']) <= 20:
        for subset, score in analysis['valid_subsets']:
            print(f"  {subset} -> score: {score}")
    else:
        print(f"  {len(analysis['valid_subsets'])} valid subsets (showing first 10):")
        for subset, score in analysis['valid_subsets'][:10]:
            print(f"  {subset} -> score: {score}")
    
    # Score distribution
    scores = [score for _, score in analysis['valid_subsets']]
    if scores:
        print(f"\nScore Distribution:")
        print(f"  Min subset score: {min(scores)}")
        print(f"  Max subset score: {max(scores)}")
        print(f"  Average subset score: {sum(scores)/len(scores):.2f}")
        
        zero_score_subsets = sum(1 for score in scores if score == 0)
        print(f"  Zero-score subsets: {zero_score_subsets}")
    
    return optimized


def minimum_incompatibility_variants():
    """
    MINIMUM INCOMPATIBILITY VARIANTS:
    ================================
    Different scenarios and modifications.
    """
    
    def maximum_incompatibility(nums, k):
        """Find maximum incompatibility instead of minimum"""
        # Modify the DP to maximize instead of minimize
        from collections import Counter
        
        n = len(nums)
        if n % k != 0:
            return -1
        
        subset_size = n // k
        count = Counter(nums)
        
        if any(freq > k for freq in count.values()):
            return -1
        
        # Generate valid subsets
        valid_subsets = []
        for mask in range(1, 1 << n):
            if bin(mask).count('1') != subset_size:
                continue
            
            subset = [nums[i] for i in range(n) if mask & (1 << i)]
            if len(set(subset)) == subset_size:
                score = max(subset) - min(subset)
                valid_subsets.append((mask, score))
        
        # DP for maximum
        dp = [-1] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == -1:
                continue
            
            for subset_mask, score in valid_subsets:
                if (mask & subset_mask) == 0:
                    new_mask = mask | subset_mask
                    dp[new_mask] = max(dp[new_mask], dp[mask] + score)
        
        return dp[(1 << n) - 1]
    
    def count_optimal_partitions(nums, k):
        """Count number of partitions achieving minimum incompatibility"""
        optimal_score = minimum_incompatibility_optimized_dp(nums, k)
        
        if optimal_score == -1:
            return 0
        
        # Count partitions with optimal score
        from collections import Counter
        
        n = len(nums)
        subset_size = n // k
        count = Counter(nums)
        
        if any(freq > k for freq in count.values()):
            return 0
        
        # Generate valid subsets
        valid_subsets = []
        for mask in range(1, 1 << n):
            if bin(mask).count('1') != subset_size:
                continue
            
            subset = [nums[i] for i in range(n) if mask & (1 << i)]
            if len(set(subset)) == subset_size:
                score = max(subset) - min(subset)
                valid_subsets.append((mask, score))
        
        # Count ways to achieve optimal score
        memo = {}
        
        def count_ways(mask, remaining_score):
            if mask == (1 << n) - 1:
                return 1 if remaining_score == 0 else 0
            
            if (mask, remaining_score) in memo:
                return memo[(mask, remaining_score)]
            
            result = 0
            for subset_mask, score in valid_subsets:
                if (mask & subset_mask) == 0 and score <= remaining_score:
                    new_mask = mask | subset_mask
                    result += count_ways(new_mask, remaining_score - score)
            
            memo[(mask, remaining_score)] = result
            return result
        
        return count_ways(0, optimal_score)
    
    def incompatibility_with_fixed_subsets(nums, k, fixed_subsets):
        """Find minimum incompatibility with some subsets pre-determined"""
        # This is more complex - simplified version
        remaining_nums = nums[:]
        
        # Remove elements in fixed subsets
        fixed_score = 0
        for subset in fixed_subsets:
            fixed_score += max(subset) - min(subset)
            for num in subset:
                if num in remaining_nums:
                    remaining_nums.remove(num)
        
        if not remaining_nums:
            return fixed_score
        
        # Solve for remaining elements
        remaining_k = k - len(fixed_subsets)
        if remaining_k <= 0:
            return -1
        
        remaining_score = minimum_incompatibility_optimized_dp(remaining_nums, remaining_k)
        
        if remaining_score == -1:
            return -1
        
        return fixed_score + remaining_score
    
    # Test variants
    test_cases = [
        ([1, 2, 1, 4], 2),
        ([6, 3, 8, 1, 3, 1, 2, 2], 4),
        ([1, 2, 3, 4], 2),
        ([1, 1, 2, 2], 2)
    ]
    
    print("Minimum Incompatibility Variants:")
    print("=" * 50)
    
    for nums, k in test_cases:
        print(f"\nNums: {nums}, k: {k}")
        
        min_incomp = minimum_incompatibility_optimized_dp(nums, k)
        print(f"Minimum incompatibility: {min_incomp}")
        
        if min_incomp != -1:
            max_incomp = maximum_incompatibility(nums, k)
            print(f"Maximum incompatibility: {max_incomp}")
            
            count_optimal = count_optimal_partitions(nums, k)
            print(f"Number of optimal partitions: {count_optimal}")
            
            # Fixed subset example
            if len(nums) >= 4 and k >= 2:
                fixed = [[nums[0], nums[1]]]
                fixed_result = incompatibility_with_fixed_subsets(nums, k, fixed)
                print(f"With fixed subset {fixed}: {fixed_result}")


# Test cases
def test_minimum_incompatibility():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 2, 1, 4], 2, 2),
        ([6, 3, 8, 1, 3, 1, 2, 2], 4, 6),
        ([5, 3, 3, 6, 3, 3], 2, -1),
        ([1, 2, 3, 4], 2, 1),
        ([1, 1, 2, 2], 2, 0),
        ([1, 2, 3, 4, 5, 6], 3, 2)
    ]
    
    print("Testing Minimum Incompatibility Solutions:")
    print("=" * 70)
    
    for i, (nums, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"nums = {nums}, k = {k}")
        print(f"Expected: {expected}")
        
        # Skip backtracking for large inputs
        if len(nums) <= 8:
            try:
                backtrack = minimum_incompatibility_backtrack(nums, k)
                print(f"Backtracking:     {backtrack:>4} {'✓' if backtrack == expected else '✗'}")
            except:
                print(f"Backtracking:     Timeout")
        
        bitmask_dp = minimum_incompatibility_bitmask_dp(nums, k)
        optimized = minimum_incompatibility_optimized_dp(nums, k)
        
        print(f"Bitmask DP:       {bitmask_dp:>4} {'✓' if bitmask_dp == expected else '✗'}")
        print(f"Optimized:        {optimized:>4} {'✓' if optimized == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    minimum_incompatibility_analysis([1, 2, 1, 4], 2)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    minimum_incompatibility_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. SUBSET CONSTRAINTS: Each subset must have unique elements")
    print("2. INCOMPATIBILITY METRIC: Range (max - min) within each subset")
    print("3. PARTITIONING OPTIMIZATION: Find optimal way to group elements")
    print("4. FEASIBILITY CHECKING: Ensure valid partition exists")
    print("5. SUBSET ENUMERATION: Generate all valid subsets efficiently")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Data Clustering: Group data points to minimize within-cluster variance")
    print("• Resource Allocation: Distribute items to minimize cost differences")
    print("• Load Balancing: Partition tasks to minimize range of completion times")
    print("• Quality Control: Group products to minimize quality variation")
    print("• Operations Research: Optimize partitioning with range constraints")


if __name__ == "__main__":
    test_minimum_incompatibility()


"""
MINIMUM INCOMPATIBILITY - SUBSET PARTITIONING WITH UNIQUENESS CONSTRAINTS:
==========================================================================

This problem demonstrates complex subset partitioning optimization:
- Partition elements into k groups with unique elements per group
- Minimize sum of ranges (max - min) across all groups
- Handle feasibility constraints and optimal substructure
- Advanced bitmask DP with constraint satisfaction

KEY INSIGHTS:
============
1. **SUBSET CONSTRAINTS**: Each subset must contain only unique elements
2. **INCOMPATIBILITY METRIC**: Range (max - min) quantifies subset diversity
3. **PARTITIONING OPTIMIZATION**: Find optimal grouping to minimize total cost
4. **FEASIBILITY CONDITIONS**: Element frequency limits and divisibility requirements
5. **SUBSET ENUMERATION**: Generate all valid subsets meeting size and uniqueness constraints

ALGORITHM APPROACHES:
====================

1. **Backtracking**: O(k^n) time, O(n) space
   - Try all possible element assignments to subsets
   - Heavy pruning needed for practical performance

2. **Bitmask DP**: O(3^n) time, O(2^n) space
   - Precompute valid subsets and use subset enumeration
   - Standard approach for exact solutions

3. **Optimized DP**: O(3^n) time, O(2^n) space
   - Memoization with subset counting optimization
   - Better practical performance

4. **Analysis Version**: O(3^n) time, O(2^n) space
   - Include detailed solution reconstruction and analysis
   - Comprehensive problem characterization

CORE SUBSET ENUMERATION DP:
===========================
```python
def minimumIncompatibility(nums, k):
    from collections import Counter
    
    n = len(nums)
    if n % k != 0:
        return -1
    
    subset_size = n // k
    count = Counter(nums)
    
    # Feasibility check
    if any(freq > k for freq in count.values()):
        return -1
    
    # Generate valid subsets
    valid_subsets = []
    for mask in range(1, 1 << n):
        if bin(mask).count('1') != subset_size:
            continue
        
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        if len(set(subset)) == subset_size:  # All unique
            score = max(subset) - min(subset)
            valid_subsets.append((mask, score))
    
    # DP: dp[mask] = min cost to partition elements in mask
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    
    for mask in range(1 << n):
        if dp[mask] == float('inf'):
            continue
        
        for subset_mask, score in valid_subsets:
            if (mask & subset_mask) == 0:  # No overlap
                new_mask = mask | subset_mask
                dp[new_mask] = min(dp[new_mask], dp[mask] + score)
    
    full_mask = (1 << n) - 1
    return dp[full_mask] if dp[full_mask] != float('inf') else -1
```

FEASIBILITY ANALYSIS:
====================
**Basic Requirements**:
- Array length must be divisible by k
- Each element can appear at most k times (pigeonhole principle)
- Must be able to form k disjoint subsets of equal size

**Mathematical Constraints**:
```python
def is_feasible(nums, k):
    n = len(nums)
    if n % k != 0:
        return False
    
    from collections import Counter
    count = Counter(nums)
    
    # Each element can appear at most k times
    if any(freq > k for freq in count.values()):
        return False
    
    return True
```

SUBSET GENERATION AND VALIDATION:
=================================
**Valid Subset Criteria**:
- Exactly n/k elements
- All elements must be unique (no duplicates)
- Incompatibility = max(subset) - min(subset)

**Efficient Generation**:
```python
def generate_valid_subsets(nums, subset_size):
    valid_subsets = []
    n = len(nums)
    
    for mask in range(1, 1 << n):
        if bin(mask).count('1') != subset_size:
            continue
        
        subset = [nums[i] for i in range(n) if mask & (1 << i)]
        if len(set(subset)) == subset_size:
            score = max(subset) - min(subset) if subset else 0
            valid_subsets.append((mask, score))
    
    return valid_subsets
```

SUBSET ENUMERATION PATTERN:
===========================
**3^n Complexity**: For each subset of elements, try all ways to partition it
- Iterate through all possible masks
- For each mask, try all valid subsets that fit
- Remaining elements form complementary subproblem

**State Transition**:
```python
for mask in range(1 << n):
    for subset_mask in valid_subsets:
        if no_overlap(mask, subset_mask):
            update_dp(mask | subset_mask, dp[mask] + subset_cost)
```

INCOMPATIBILITY METRIC:
======================
**Range Calculation**: For subset S, incompatibility = max(S) - min(S)
- Minimizes when elements are close in value
- Zero when all elements identical
- Grows with diversity within subset

**Optimization Objective**: Minimize sum of ranges across all subsets
```python
total_incompatibility = sum(max(subset) - min(subset) for subset in partition)
```

CONSTRAINT SATISFACTION:
=======================
**Hard Constraints**:
- Exactly k non-empty subsets
- Each subset has exactly n/k elements  
- No element appears twice in same subset
- Every element assigned to exactly one subset

**Soft Objective**: Minimize total incompatibility

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(3^n)
- Generate O(2^n) possible subsets
- For each state, try O(2^n) subset choices
- Subset enumeration creates 3^n pattern

**Space Complexity**: O(2^n)
- DP table indexed by element bitmasks
- Additional space for valid subset storage

**Practical Limits**: n ≤ 16-20 due to exponential nature

OPTIMIZATION TECHNIQUES:
=======================
**Subset Precomputation**: Generate valid subsets once
**Feasibility Pruning**: Early termination for impossible cases
**State Compression**: Efficient bitmask operations
**Memoization**: Cache computed subproblems

PATH RECONSTRUCTION:
===================
**Parent Tracking**: Store optimal subset choice for each state
```python
parent[new_mask] = (prev_mask, chosen_subset, subset_cost)
```

**Solution Recovery**: Backtrack through parent pointers
```python
def reconstruct_partition():
    partition = []
    mask = full_mask
    
    while mask != 0:
        prev_mask, subset, cost = parent[mask]
        partition.append((subset, cost))
        mask = prev_mask
    
    return partition
```

APPLICATIONS:
============
- **Data Clustering**: Group data points to minimize within-cluster variance
- **Resource Allocation**: Distribute resources to minimize cost variation
- **Load Balancing**: Partition tasks to minimize completion time ranges
- **Quality Control**: Group items to minimize quality variation
- **Tournament Design**: Create balanced groups with diverse skill levels

RELATED PROBLEMS:
================
- **Balanced Partition**: Minimize difference between subset sums
- **K-means Clustering**: Minimize within-cluster sum of squares
- **Bin Packing**: Pack items to minimize number of bins
- **Graph Coloring**: Assign colors with constraints

VARIANTS:
========
- **Maximum Incompatibility**: Maximize instead of minimize ranges
- **Weighted Elements**: Elements have different importance weights
- **Size Flexibility**: Allow variable subset sizes
- **Multiple Objectives**: Balance incompatibility with other metrics

EDGE CASES:
==========
- **Impossible Partitions**: Too many duplicate elements
- **Zero Incompatibility**: All elements identical within subsets
- **Single Element Subsets**: k = n case
- **Minimum k**: k = 1 (all elements in one subset)

This problem showcases advanced constraint satisfaction
in combinatorial optimization, demonstrating how complex
feasibility conditions and optimization objectives can
be handled through systematic subset enumeration and
dynamic programming techniques.
"""
