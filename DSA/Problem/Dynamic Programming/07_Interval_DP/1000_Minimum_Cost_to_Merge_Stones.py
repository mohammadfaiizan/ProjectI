"""
LeetCode 1000: Minimum Cost to Merge Stones
Difficulty: Hard
Category: Interval DP - Generalized Merging

PROBLEM DESCRIPTION:
===================
There are n piles of stones arranged in a row. The ith pile has stones[i] stones.
A move consists of merging exactly k consecutive piles into one pile, and the cost of this move is equal to the total number of stones in these k piles.
Return the minimum cost to merge all piles into one pile. If it is impossible, return -1.

Example 1:
Input: stones = [3,2,4,1], k = 2
Output: 20
Explanation: We start with [3, 2, 4, 1].
We merge [3, 2] for a cost of 5, and we are left with [5, 4, 1].
We merge [5, 4] for a cost of 9, and we are left with [9, 1].
We merge [9, 1] for a cost of 10, and we are left with [10].
The total cost was 20, and this is the minimum possible.

Example 2:
Input: stones = [3,2,4,1], k = 3
Output: -1
Explanation: After any number of moves, there will always be at least 2 piles left, and we cannot merge anymore.

Example 3:
Input: stones = [3,5,1,2,6], k = 3
Output: 25
Explanation: We start with [3, 5, 1, 2, 6].
We merge [5, 1, 2] for a cost of 8, and we are left with [3, 8, 6].
We merge [3, 8, 6] for a cost of 17, and we are left with [17].
The total cost was 25, and this is the minimum possible.

Constraints:
- n == stones.length
- 1 <= n <= 30
- 1 <= stones[i] <= 100
- 2 <= k <= 30
"""

def merge_stones_brute_force(stones, k):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible merging sequences recursively.
    
    Time Complexity: O(exponential) - explosive branching
    Space Complexity: O(n) - recursion depth
    """
    n = len(stones)
    
    # Check if merging is possible
    if (n - 1) % (k - 1) != 0:
        return -1
    
    def merge_recursive(piles):
        if len(piles) == 1:
            return 0
        
        min_cost = float('inf')
        
        # Try all possible consecutive k-pile merges
        for i in range(len(piles) - k + 1):
            # Cost of merging k consecutive piles starting at i
            merge_cost = sum(piles[i:i+k])
            
            # Create new pile configuration
            new_piles = piles[:i] + [merge_cost] + piles[i+k:]
            
            # Recurse with new configuration
            remaining_cost = merge_recursive(new_piles)
            if remaining_cost != -1:
                min_cost = min(min_cost, merge_cost + remaining_cost)
        
        return min_cost if min_cost != float('inf') else -1
    
    return merge_recursive(stones)


def merge_stones_memoization(stones, k):
    """
    MEMOIZATION APPROACH:
    ====================
    Cache results for different pile configurations.
    
    Time Complexity: O(n^3 / k) - with memoization
    Space Complexity: O(n^3) - memo table
    """
    n = len(stones)
    
    if (n - 1) % (k - 1) != 0:
        return -1
    
    # Precompute prefix sums for range sum queries
    prefix_sum = [0]
    for stone in stones:
        prefix_sum.append(prefix_sum[-1] + stone)
    
    def range_sum(i, j):
        return prefix_sum[j + 1] - prefix_sum[i]
    
    memo = {}
    
    def dp(left, right, piles):
        """
        left, right: range [left, right]
        piles: number of piles to merge this range into
        Returns: minimum cost to merge range into 'piles' piles
        """
        if (left, right, piles) in memo:
            return memo[(left, right, piles)]
        
        if left == right:
            result = 0 if piles == 1 else float('inf')
        elif piles == 1:
            # To merge into 1 pile, first merge into k piles, then merge those k piles
            temp = dp(left, right, k)
            if temp == float('inf'):
                result = float('inf')
            else:
                result = temp + range_sum(left, right)
        else:
            # Merge into 'piles' piles by splitting optimally
            result = float('inf')
            for mid in range(left, right, k - 1):
                left_cost = dp(left, mid, 1)
                right_cost = dp(mid + 1, right, piles - 1)
                
                if left_cost != float('inf') and right_cost != float('inf'):
                    result = min(result, left_cost + right_cost)
        
        memo[(left, right, piles)] = result
        return result
    
    result = dp(0, n - 1, 1)
    return result if result != float('inf') else -1


def merge_stones_interval_dp(stones, k):
    """
    INTERVAL DP APPROACH:
    ====================
    Bottom-up DP with range processing.
    
    Time Complexity: O(n^3 / k) - three nested loops with k-step increment
    Space Complexity: O(n^3) - 3D DP table
    """
    n = len(stones)
    
    if (n - 1) % (k - 1) != 0:
        return -1
    
    # Prefix sums for efficient range sum calculation
    prefix_sum = [0]
    for stone in stones:
        prefix_sum.append(prefix_sum[-1] + stone)
    
    def range_sum(i, j):
        return prefix_sum[j + 1] - prefix_sum[i]
    
    # dp[i][j][m] = min cost to merge stones[i:j+1] into m piles
    dp = [[[float('inf')] * (k + 1) for _ in range(n)] for _ in range(n)]
    
    # Base case: single stone is already 1 pile with 0 cost
    for i in range(n):
        dp[i][i][1] = 0
    
    # Process by range length
    for length in range(2, n + 1):  # Range length from 2 to n
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Try to merge into m piles (2 <= m <= k)
            for m in range(2, k + 1):
                for mid in range(i, j, k - 1):
                    dp[i][j][m] = min(dp[i][j][m], 
                                     dp[i][mid][1] + dp[mid + 1][j][m - 1])
            
            # Merge k piles into 1 pile
            dp[i][j][1] = dp[i][j][k] + range_sum(i, j)
    
    return dp[0][n - 1][1] if dp[0][n - 1][1] != float('inf') else -1


def merge_stones_optimized_dp(stones, k):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Space-optimized version with better constants.
    
    Time Complexity: O(n^3 / k) - same asymptotic complexity
    Space Complexity: O(n^2) - 2D DP table
    """
    n = len(stones)
    
    if (n - 1) % (k - 1) != 0:
        return -1
    
    # Prefix sums
    prefix_sum = [0]
    for stone in stones:
        prefix_sum.append(prefix_sum[-1] + stone)
    
    def range_sum(i, j):
        return prefix_sum[j + 1] - prefix_sum[i]
    
    # dp[i][j] = minimum cost to merge stones[i:j+1] into minimum possible piles
    dp = [[float('inf')] * n for _ in range(n)]
    
    # Base case
    for i in range(n):
        dp[i][i] = 0
    
    # Process by range length
    for length in range(k, n + 1):  # Start from k since we need at least k piles to merge
        for i in range(n - length + 1):
            j = i + length - 1
            
            # Try all possible split points
            for mid in range(i, j, k - 1):
                dp[i][j] = min(dp[i][j], dp[i][mid] + dp[mid + 1][j])
            
            # If this range can be merged into 1 pile, add the merge cost
            if (j - i) % (k - 1) == 0:
                dp[i][j] += range_sum(i, j)
    
    return dp[0][n - 1] if dp[0][n - 1] != float('inf') else -1


def merge_stones_with_sequence(stones, k):
    """
    TRACK MERGING SEQUENCE:
    ======================
    Return minimum cost and the actual sequence of merges.
    
    Time Complexity: O(n^3 / k) - DP computation + reconstruction
    Space Complexity: O(n^3) - DP table + sequence tracking
    """
    n = len(stones)
    
    if (n - 1) % (k - 1) != 0:
        return -1, []
    
    prefix_sum = [0]
    for stone in stones:
        prefix_sum.append(prefix_sum[-1] + stone)
    
    def range_sum(i, j):
        return prefix_sum[j + 1] - prefix_sum[i]
    
    dp = [[[float('inf')] * (k + 1) for _ in range(n)] for _ in range(n)]
    choice = [[[None] * (k + 1) for _ in range(n)] for _ in range(n)]
    
    # Base case
    for i in range(n):
        dp[i][i][1] = 0
    
    # Fill DP table
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            for m in range(2, k + 1):
                for mid in range(i, j, k - 1):
                    cost = dp[i][mid][1] + dp[mid + 1][j][m - 1]
                    if cost < dp[i][j][m]:
                        dp[i][j][m] = cost
                        choice[i][j][m] = mid
            
            if dp[i][j][k] != float('inf'):
                dp[i][j][1] = dp[i][j][k] + range_sum(i, j)
                choice[i][j][1] = -1  # Indicates final merge
    
    # Reconstruct merging sequence
    def get_merges(left, right, piles):
        if left == right:
            return []
        
        merges = []
        
        if piles == 1:
            # Final merge from k piles to 1
            sub_merges = get_merges(left, right, k)
            merges.extend(sub_merges)
            merges.append((left, right, range_sum(left, right)))
        else:
            # Split into smaller problems
            mid = choice[left][right][piles]
            merges.extend(get_merges(left, mid, 1))
            merges.extend(get_merges(mid + 1, right, piles - 1))
        
        return merges
    
    min_cost = dp[0][n - 1][1]
    if min_cost == float('inf'):
        return -1, []
    
    sequence = get_merges(0, n - 1, 1)
    return min_cost, sequence


def merge_stones_analysis(stones, k):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and merging analysis.
    """
    print(f"Minimum Cost to Merge Stones Analysis:")
    print(f"Stones: {stones}")
    print(f"Merge parameter k: {k}")
    print(f"Number of stones: {len(stones)}")
    
    n = len(stones)
    
    # Check feasibility
    print(f"\nFeasibility check:")
    print(f"To merge n piles into 1 pile with k-way merging:")
    print(f"Each merge reduces pile count by (k-1)")
    print(f"Need (n-1) to be divisible by (k-1)")
    print(f"(n-1) % (k-1) = ({n}-1) % ({k}-1) = {(n-1) % (k-1)}")
    
    if (n - 1) % (k - 1) != 0:
        print(f"❌ Impossible to merge all stones into one pile!")
        return -1
    else:
        print(f"✅ Merging is possible!")
    
    # Show prefix sums
    prefix_sum = [0]
    for stone in stones:
        prefix_sum.append(prefix_sum[-1] + stone)
    
    print(f"\nPrefix sums: {prefix_sum}")
    print(f"Range sum examples:")
    for i in range(min(3, n)):
        for j in range(i + k - 1, min(i + k + 2, n)):
            if j < n:
                range_cost = prefix_sum[j + 1] - prefix_sum[i]
                print(f"  stones[{i}:{j+1}] = {stones[i:j+1]} → cost = {range_cost}")
    
    # Build DP table with logging
    dp = [[[float('inf')] * (k + 1) for _ in range(n)] for _ in range(n)]
    
    # Base case
    print(f"\nBase case (single stones):")
    for i in range(n):
        dp[i][i][1] = 0
        print(f"  dp[{i}][{i}][1] = 0 (stone {stones[i]} is already 1 pile)")
    
    print(f"\nDP computation:")
    
    for length in range(2, min(n + 1, 6)):  # Show first few lengths
        print(f"\nLength {length} ranges:")
        for i in range(n - length + 1):
            j = i + length - 1
            
            print(f"  Range [{i},{j}]: stones {stones[i:j+1]}")
            
            # Try different pile counts
            for m in range(2, min(k + 1, length + 1)):
                best_cost = float('inf')
                best_split = -1
                
                for mid in range(i, j, k - 1):
                    if mid < j:
                        left_cost = dp[i][mid][1]
                        right_cost = dp[mid + 1][j][m - 1]
                        
                        if left_cost != float('inf') and right_cost != float('inf'):
                            cost = left_cost + right_cost
                            if cost < best_cost:
                                best_cost = cost
                                best_split = mid
                
                if best_cost != float('inf'):
                    dp[i][j][m] = best_cost
                    print(f"    Into {m} piles: cost {best_cost} (split at {best_split})")
            
            # Try merging into 1 pile
            if dp[i][j][k] != float('inf'):
                merge_cost = prefix_sum[j + 1] - prefix_sum[i]
                dp[i][j][1] = dp[i][j][k] + merge_cost
                print(f"    Into 1 pile: {dp[i][j][k]} + {merge_cost} = {dp[i][j][1]}")
    
    result = dp[0][n - 1][1]
    print(f"\nMinimum cost: {result}")
    
    # Show actual merging sequence
    min_cost, sequence = merge_stones_with_sequence(stones, k)
    if sequence:
        print(f"\nOptimal merging sequence:")
        total_cost = 0
        for i, (left, right, cost) in enumerate(sequence):
            total_cost += cost
            print(f"  Step {i + 1}: Merge range [{left},{right}] with cost {cost}")
            print(f"    Total cost so far: {total_cost}")
    
    return result


def merge_stones_variants():
    """
    MERGE STONES VARIANTS:
    =====================
    Different scenarios and modifications.
    """
    
    def merge_stones_max_cost(stones, k):
        """Find maximum cost to merge stones"""
        n = len(stones)
        
        if (n - 1) % (k - 1) != 0:
            return -1
        
        prefix_sum = [0]
        for stone in stones:
            prefix_sum.append(prefix_sum[-1] + stone)
        
        def range_sum(i, j):
            return prefix_sum[j + 1] - prefix_sum[i]
        
        dp = [[[0] * (k + 1) for _ in range(n)] for _ in range(n)]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                for m in range(2, k + 1):
                    for mid in range(i, j, k - 1):
                        dp[i][j][m] = max(dp[i][j][m], 
                                         dp[i][mid][1] + dp[mid + 1][j][m - 1])
                
                dp[i][j][1] = dp[i][j][k] + range_sum(i, j)
        
        return dp[0][n - 1][1]
    
    def count_merge_ways(stones, k):
        """Count number of ways to merge stones"""
        # This is complex to implement efficiently
        # For demonstration, return 1 if possible, 0 otherwise
        n = len(stones)
        return 1 if (n - 1) % (k - 1) == 0 else 0
    
    def merge_with_different_k(stones, k_values):
        """Try merging with different k values"""
        results = {}
        for k in k_values:
            cost = merge_stones_interval_dp(stones, k)
            results[k] = cost
        return results
    
    # Test variants
    test_cases = [
        ([3, 2, 4, 1], 2),
        ([3, 2, 4, 1], 3),
        ([3, 5, 1, 2, 6], 3),
        ([1, 2, 3, 4, 5], 2),
        ([1, 2, 3, 4, 5, 6], 3)
    ]
    
    print("Merge Stones Variants:")
    print("=" * 50)
    
    for stones, k in test_cases:
        print(f"\nStones: {stones}, k = {k}")
        
        min_cost = merge_stones_interval_dp(stones, k)
        print(f"Min cost: {min_cost}")
        
        if min_cost != -1:
            max_cost = merge_stones_max_cost(stones, k)
            print(f"Max cost: {max_cost}")
        
        # Try different k values
        k_values = [2, 3, 4]
        k_results = merge_with_different_k(stones, k_values)
        print(f"Different k values: {k_results}")


# Test cases
def test_merge_stones():
    """Test all implementations with various inputs"""
    test_cases = [
        ([3, 2, 4, 1], 2, 20),
        ([3, 2, 4, 1], 3, -1),
        ([3, 5, 1, 2, 6], 3, 25),
        ([1, 2, 3, 4, 5], 2, 30),
        ([1, 2], 2, 3),
        ([1], 2, 0),
        ([1, 4, 3, 2, 5], 3, 22),
        ([6, 4, 4, 6], 2, 40)
    ]
    
    print("Testing Minimum Cost to Merge Stones Solutions:")
    print("=" * 70)
    
    for i, (stones, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Stones: {stones}, k = {k}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large inputs
        if len(stones) <= 5:
            try:
                brute_force = merge_stones_brute_force(stones, k)
                print(f"Brute Force:      {brute_force:>4} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memoization = merge_stones_memoization(stones, k)
        interval_dp = merge_stones_interval_dp(stones, k)
        optimized_dp = merge_stones_optimized_dp(stones, k)
        
        print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        print(f"Interval DP:      {interval_dp:>4} {'✓' if interval_dp == expected else '✗'}")
        print(f"Optimized DP:     {optimized_dp:>4} {'✓' if optimized_dp == expected else '✗'}")
        
        # Show merging sequence for small cases
        if len(stones) <= 6 and expected != -1:
            min_cost, sequence = merge_stones_with_sequence(stones, k)
            print(f"Merging sequence: {len(sequence)} steps")
            if len(sequence) <= 5:
                for j, (left, right, cost) in enumerate(sequence):
                    print(f"  Step {j+1}: Range [{left},{right}] cost {cost}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    merge_stones_analysis([3, 5, 1, 2, 6], 3)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    merge_stones_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. FEASIBILITY: (n-1) must be divisible by (k-1)")
    print("2. GENERALIZED MERGING: Extends burst balloons to k-way merging")
    print("3. 3D DP: dp[i][j][m] for merging range into m piles")
    print("4. RANGE SPLITTING: Split at positions that are multiples of (k-1)")
    print("5. MERGE COST: Sum of all stones in the merged range")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Resource Management: Optimal batching strategies")
    print("• Parallel Processing: Task merging optimization")
    print("• Data Structures: Heap merging and tree construction")
    print("• Algorithm Design: Generalized interval merging")
    print("• Operations Research: Batch processing optimization")


if __name__ == "__main__":
    test_merge_stones()


"""
MINIMUM COST TO MERGE STONES - GENERALIZED INTERVAL MERGING:
============================================================

This problem generalizes burst balloons to k-way merging:
- Can only merge exactly k consecutive piles at a time
- Cost equals sum of stones in merged piles
- Must determine if merging all stones into one pile is possible
- Demonstrates advanced interval DP with feasibility constraints

KEY INSIGHTS:
============
1. **FEASIBILITY CONDITION**: (n-1) must be divisible by (k-1)
2. **K-WAY MERGING**: Each merge operation combines exactly k piles
3. **3D DP STATE**: dp[i][j][m] = min cost to merge range [i,j] into m piles
4. **RANGE SPLITTING**: Split points must be at multiples of (k-1)
5. **GENERALIZATION**: Extends classic interval DP to multi-way operations

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(exponential) time, O(n) space
   - Try all possible merging sequences
   - Only viable for tiny inputs

2. **Memoization**: O(n³/k) time, O(n³) space
   - Top-down DP with 3D state caching
   - Natural recursive structure

3. **Interval DP**: O(n³/k) time, O(n³) space
   - Bottom-up 3D DP construction
   - Standard approach for this problem

4. **Optimized DP**: O(n³/k) time, O(n²) space
   - Space-optimized version
   - Better practical performance

FEASIBILITY ANALYSIS:
====================
**Why (n-1) % (k-1) == 0 is necessary**:
- Start with n piles
- Each merge reduces pile count by (k-1)
- After m merges: n - m×(k-1) piles remain
- To reach 1 pile: n - m×(k-1) = 1
- Therefore: m = (n-1)/(k-1)
- This must be an integer

**Examples**:
- n=4, k=2: (4-1)%(2-1) = 3%1 = 0 ✓ (need 3 merges)
- n=4, k=3: (4-1)%(3-1) = 3%2 = 1 ✗ (impossible)
- n=5, k=3: (5-1)%(3-1) = 4%2 = 0 ✓ (need 2 merges)

CORE 3D DP ALGORITHM:
====================
```python
# dp[i][j][m] = min cost to merge stones[i:j+1] into m piles
dp = [[[inf] * (k+1) for _ in range(n)] for _ in range(n)]

# Base case: single stone is 1 pile with 0 cost
for i in range(n):
    dp[i][i][1] = 0

for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        
        # Merge into m piles (2 ≤ m ≤ k)
        for m in range(2, k + 1):
            for mid in range(i, j, k - 1):  # Split at k-1 intervals
                dp[i][j][m] = min(dp[i][j][m], 
                                 dp[i][mid][1] + dp[mid+1][j][m-1])
        
        # Merge k piles into 1 pile
        dp[i][j][1] = dp[i][j][k] + sum(stones[i:j+1])
```

RECURRENCE RELATIONS:
====================
```
For m ≥ 2:
dp[i][j][m] = min(dp[i][mid][1] + dp[mid+1][j][m-1])
              for mid in range(i, j, k-1)

For m = 1:
dp[i][j][1] = dp[i][j][k] + sum(stones[i:j+1])

Base case:
dp[i][i][1] = 0  (single stone is already 1 pile)
```

**Intuition**:
- To merge range [i,j] into m piles: split optimally and merge recursively
- To merge range [i,j] into 1 pile: first get k piles, then merge them
- Split points must be at k-1 intervals to maintain feasibility

SPLIT POINT CONSTRAINTS:
========================
**Why split at multiples of (k-1)**:
- If we split range [i,j] at position mid
- Left part [i,mid] → 1 pile
- Right part [mid+1,j] → (m-1) piles
- For right part to be feasible: (j-mid-1+1-1) % (k-1) == 0
- This gives: (j-mid-1) % (k-1) == 0
- Therefore: mid = i + t×(k-1) for integer t

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n³/k) - three nested loops, inner loop steps by k-1
- **Space**: O(n³) for 3D DP, O(n²) optimized
- **States**: O(n²×k) - all intervals with all pile counts
- **Transitions**: O(n/k) - split points at k-1 intervals

SPACE OPTIMIZATION:
==================
Since we only need dp[i][j][k] to compute dp[i][j][1]:
```python
# 2D version focusing on minimum piles possible
dp = [[inf] * n for _ in range(n)]

for i in range(n):
    dp[i][i] = 0

for length in range(k, n + 1):  # Need at least k stones to merge
    for i in range(n - length + 1):
        j = i + length - 1
        
        # Try all valid split points
        for mid in range(i, j, k - 1):
            dp[i][j] = min(dp[i][j], dp[i][mid] + dp[mid+1][j])
        
        # Add merge cost if this range can become 1 pile
        if (j - i) % (k - 1) == 0:
            dp[i][j] += sum(stones[i:j+1])
```

SOLUTION RECONSTRUCTION:
=======================
To find actual merging sequence:
```python
def get_merges(left, right, piles):
    if left == right:
        return []
    
    merges = []
    if piles == 1:
        # First get k piles, then merge them
        sub_merges = get_merges(left, right, k)
        merges.extend(sub_merges)
        merges.append((left, right, sum(stones[left:right+1])))
    else:
        # Split into subproblems
        mid = choice[left][right][piles]
        merges.extend(get_merges(left, mid, 1))
        merges.extend(get_merges(mid+1, right, piles-1))
    
    return merges
```

APPLICATIONS:
============
- **Resource Management**: Optimal batching strategies
- **Parallel Processing**: Task merging and load balancing
- **Data Structures**: Multi-way heap merging
- **Operations Research**: Batch processing optimization
- **Algorithm Design**: Generalized interval optimization

RELATED PROBLEMS:
================
- **Burst Balloons (312)**: Special case with k=2 (binary merging)
- **Matrix Chain Multiplication**: Similar interval structure
- **Optimal Binary Search Trees**: Weighted interval optimization
- **Huffman Coding**: Optimal merging for compression

MATHEMATICAL PROPERTIES:
========================
- **Feasibility**: Determined by divisibility condition
- **Optimal Substructure**: Optimal solution contains optimal subsolutions
- **Overlapping Subproblems**: Same ranges computed multiple times
- **Monotonicity**: More merging options never increase minimum cost

EDGE CASES:
==========
- **Single stone**: Cost = 0 (already merged)
- **Impossible cases**: Return -1 when (n-1) % (k-1) ≠ 0
- **k = 2**: Reduces to binary merging (similar to burst balloons)
- **k ≥ n**: Only one merge operation possible

This problem beautifully generalizes interval DP to multi-way operations,
showing how constraints on operation types can dramatically change both
the algorithm structure and the feasibility conditions.
"""
