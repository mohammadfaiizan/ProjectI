"""
LeetCode 1478: Allocate Mailboxes
Difficulty: Hard
Category: Grid/Matrix DP

PROBLEM DESCRIPTION:
===================
Given the array houses and an integer k. where houses[i] is the location of the ith house along a street, 
your task is to allocate k mailboxes in the street.

Return the minimum total distance between each house and its nearest mailbox.

The answer is guaranteed to fit in a 32-bit signed integer.

Example 1:
Input: houses = [1,4,8,10,20], k = 3
Output: 5
Explanation: Allocate mailboxes in position 3, 9 and 20.
Minimum total distance from each house to nearest mailbox is |1-3| + |4-3| + |8-9| + |10-9| + |20-20| = 2 + 1 + 1 + 1 + 0 = 5.

Example 2:
Input: houses = [2,3,5,12,18], k = 2
Output: 9
Explanation: Allocate mailboxes in position 3 and 14.
Minimum total distance is |2-3| + |3-3| + |5-3| + |12-14| + |18-14| = 1 + 0 + 2 + 2 + 4 = 9.

Example 3:
Input: houses = [7,4,6,1], k = 1
Output: 8

Example 4:
Input: houses = [3,6,14,10], k = 4
Output: 0

Constraints:
- n == houses.length
- 1 <= n <= 100
- 1 <= houses[i] <= 10000
- 1 <= k <= n
- Array houses contain unique integers.
"""

def min_distance_brute_force(houses, k):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible ways to group houses and place mailboxes.
    
    Time Complexity: O(S(n,k) * n^k) where S(n,k) is Stirling number of second kind
    Space Complexity: O(n) - recursion depth
    """
    houses.sort()
    n = len(houses)
    
    def optimal_mailbox_cost(start, end):
        """Cost of placing one mailbox optimally for houses[start:end+1]"""
        # Optimal position is median
        length = end - start + 1
        median_pos = (start + end) // 2
        mailbox_position = houses[median_pos]
        
        total_cost = 0
        for i in range(start, end + 1):
            total_cost += abs(houses[i] - mailbox_position)
        
        return total_cost
    
    def backtrack(house_idx, mailboxes_left):
        # Base cases
        if mailboxes_left == 0:
            return float('inf') if house_idx < n else 0
        
        if house_idx >= n:
            return float('inf') if mailboxes_left > 0 else 0
        
        if mailboxes_left == 1:
            # Use one mailbox for all remaining houses
            return optimal_mailbox_cost(house_idx, n - 1)
        
        min_cost = float('inf')
        
        # Try different ending positions for current mailbox
        for end_idx in range(house_idx, n - mailboxes_left + 1):
            # Current mailbox covers houses[house_idx:end_idx+1]
            current_cost = optimal_mailbox_cost(house_idx, end_idx)
            remaining_cost = backtrack(end_idx + 1, mailboxes_left - 1)
            
            min_cost = min(min_cost, current_cost + remaining_cost)
        
        return min_cost
    
    return backtrack(0, k)


def min_distance_memoization(houses, k):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(n^2 * k) - each state computed once
    Space Complexity: O(n * k) - memoization table
    """
    houses.sort()
    n = len(houses)
    memo = {}
    
    def optimal_mailbox_cost(start, end):
        """Cost of placing one mailbox optimally for houses[start:end+1]"""
        median_pos = (start + end) // 2
        mailbox_position = houses[median_pos]
        
        total_cost = 0
        for i in range(start, end + 1):
            total_cost += abs(houses[i] - mailbox_position)
        
        return total_cost
    
    def dp(house_idx, mailboxes_left):
        if (house_idx, mailboxes_left) in memo:
            return memo[(house_idx, mailboxes_left)]
        
        # Base cases
        if mailboxes_left == 0:
            result = float('inf') if house_idx < n else 0
        elif house_idx >= n:
            result = float('inf') if mailboxes_left > 0 else 0
        elif mailboxes_left == 1:
            result = optimal_mailbox_cost(house_idx, n - 1)
        else:
            result = float('inf')
            
            # Try different ending positions for current mailbox
            for end_idx in range(house_idx, n - mailboxes_left + 1):
                current_cost = optimal_mailbox_cost(house_idx, end_idx)
                remaining_cost = dp(end_idx + 1, mailboxes_left - 1)
                result = min(result, current_cost + remaining_cost)
        
        memo[(house_idx, mailboxes_left)] = result
        return result
    
    return dp(0, k)


def min_distance_dp_optimized(houses, k):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Precompute costs and use 2D DP table.
    
    Time Complexity: O(n^2 + n^2*k) = O(n^2*k) - precompute costs + DP
    Space Complexity: O(n^2 + n*k) = O(n^2) - cost matrix + DP table
    """
    houses.sort()
    n = len(houses)
    
    # Precompute cost[i][j] = minimum cost to serve houses[i:j+1] with one mailbox
    cost = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            # Optimal mailbox position is median
            median_pos = (i + j) // 2
            mailbox_position = houses[median_pos]
            
            total_cost = 0
            for idx in range(i, j + 1):
                total_cost += abs(houses[idx] - mailbox_position)
            
            cost[i][j] = total_cost
    
    # dp[i][m] = minimum cost to allocate m mailboxes for first i houses
    dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
    
    # Base case: 0 houses, 0 mailboxes
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, n + 1):
        for m in range(1, min(i, k) + 1):
            # Try all possible positions for the last mailbox
            for prev_houses in range(m - 1, i):
                if dp[prev_houses][m - 1] != float('inf'):
                    dp[i][m] = min(dp[i][m], 
                                 dp[prev_houses][m - 1] + cost[prev_houses][i - 1])
    
    return dp[n][k]


def min_distance_space_optimized(houses, k):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(n) space for DP computation.
    
    Time Complexity: O(n^2*k) - same as optimized DP
    Space Complexity: O(n^2) - cost matrix only (no DP table storage)
    """
    houses.sort()
    n = len(houses)
    
    # Precompute costs
    cost = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            median_pos = (i + j) // 2
            mailbox_position = houses[median_pos]
            
            total_cost = 0
            for idx in range(i, j + 1):
                total_cost += abs(houses[idx] - mailbox_position)
            
            cost[i][j] = total_cost
    
    # Use rolling arrays for DP
    prev_dp = [float('inf')] * (n + 1)
    prev_dp[0] = 0
    
    for m in range(1, k + 1):
        curr_dp = [float('inf')] * (n + 1)
        
        for i in range(m, n + 1):
            # Try all possible positions for the last mailbox
            for prev_houses in range(m - 1, i):
                if prev_dp[prev_houses] != float('inf'):
                    curr_dp[i] = min(curr_dp[i], 
                                   prev_dp[prev_houses] + cost[prev_houses][i - 1])
        
        prev_dp = curr_dp
    
    return prev_dp[n]


def min_distance_with_positions(houses, k):
    """
    FIND ACTUAL MAILBOX POSITIONS:
    ==============================
    Return minimum cost and actual optimal mailbox positions.
    
    Time Complexity: O(n^2*k) - DP + backtracking
    Space Complexity: O(n^2 + n*k) - cost matrix + DP table + parent tracking
    """
    houses.sort()
    n = len(houses)
    
    # Precompute costs and optimal positions
    cost = [[0] * n for _ in range(n)]
    mailbox_pos = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            median_pos = (i + j) // 2
            optimal_position = houses[median_pos]
            mailbox_pos[i][j] = optimal_position
            
            total_cost = 0
            for idx in range(i, j + 1):
                total_cost += abs(houses[idx] - optimal_position)
            
            cost[i][j] = total_cost
    
    # DP with parent tracking
    dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
    parent = [[-1] * (k + 1) for _ in range(n + 1)]
    
    dp[0][0] = 0
    
    for i in range(1, n + 1):
        for m in range(1, min(i, k) + 1):
            for prev_houses in range(m - 1, i):
                if dp[prev_houses][m - 1] != float('inf'):
                    new_cost = dp[prev_houses][m - 1] + cost[prev_houses][i - 1]
                    if new_cost < dp[i][m]:
                        dp[i][m] = new_cost
                        parent[i][m] = prev_houses
    
    # Reconstruct mailbox positions
    mailbox_positions = []
    i, m = n, k
    
    while m > 0 and i > 0:
        prev_i = parent[i][m]
        # Mailbox for houses[prev_i:i] is at mailbox_pos[prev_i][i-1]
        mailbox_positions.append(mailbox_pos[prev_i][i - 1])
        i, m = prev_i, m - 1
    
    mailbox_positions.reverse()
    
    return dp[n][k], mailbox_positions


def min_distance_analysis(houses, k):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and mailbox placement analysis.
    """
    print(f"House positions: {houses}")
    print(f"Number of mailboxes: {k}")
    
    houses_sorted = sorted(houses)
    n = len(houses_sorted)
    
    print(f"Sorted houses: {houses_sorted}")
    
    # Show optimal single mailbox costs for all segments
    print(f"\nOptimal single mailbox costs for all segments:")
    
    for i in range(n):
        for j in range(i, min(i + 5, n)):  # Show first few for brevity
            median_pos = (i + j) // 2
            mailbox_position = houses_sorted[median_pos]
            
            segment_houses = houses_sorted[i:j+1]
            total_cost = sum(abs(house - mailbox_position) for house in segment_houses)
            
            print(f"  Houses {segment_houses}: mailbox at {mailbox_position}, cost = {total_cost}")
    
    if n > 5:
        print(f"  ... (showing first few segments only)")
    
    # Compute optimal solution
    min_cost, positions = min_distance_with_positions(houses, k)
    
    print(f"\nOptimal Solution:")
    print(f"Minimum total distance: {min_cost}")
    print(f"Mailbox positions: {positions}")
    
    # Show house-to-mailbox assignments
    print(f"\nHouse assignments:")
    for house in houses_sorted:
        # Find nearest mailbox
        nearest_mailbox = min(positions, key=lambda pos: abs(house - pos))
        distance = abs(house - nearest_mailbox)
        print(f"  House at {house} → Mailbox at {nearest_mailbox} (distance: {distance})")
    
    # Verify total cost
    total_verification = sum(min(abs(house - pos) for pos in positions) for house in houses_sorted)
    print(f"\nCost verification: {total_verification}")
    
    return min_cost


def min_distance_complexity_analysis():
    """
    COMPLEXITY ANALYSIS:
    ===================
    Analyze how different approaches scale with input size.
    """
    import time
    import random
    
    print("Complexity Analysis for Mailbox Allocation:")
    print("=" * 60)
    
    sizes = [10, 20, 30, 40, 50]
    k_ratios = [0.2, 0.5, 0.8]  # k as fraction of n
    
    for n in sizes:
        print(f"\nHouses: {n}")
        
        # Generate random test case
        houses = sorted(random.sample(range(1, 1000), n))
        
        for ratio in k_ratios:
            k = max(1, int(n * ratio))
            print(f"  k = {k} ({ratio*100:.0f}% of houses):")
            
            # Test memoization approach
            start_time = time.time()
            result = min_distance_memoization(houses, k)
            memo_time = time.time() - start_time
            
            # Test optimized DP approach
            start_time = time.time()
            result_opt = min_distance_dp_optimized(houses, k)
            opt_time = time.time() - start_time
            
            # Test space optimized approach
            start_time = time.time()
            result_space = min_distance_space_optimized(houses, k)
            space_time = time.time() - start_time
            
            print(f"    Memoization:    {memo_time:.4f}s (result: {result})")
            print(f"    DP Optimized:   {opt_time:.4f}s (result: {result_opt})")
            print(f"    Space Opt:      {space_time:.4f}s (result: {result_space})")
            
            # Verify results match
            if not (result == result_opt == result_space):
                print(f"    WARNING: Results don't match!")


# Test cases
def test_min_distance():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 4, 8, 10, 20], 3, 5),
        ([2, 3, 5, 12, 18], 2, 9),
        ([7, 4, 6, 1], 1, 8),
        ([3, 6, 14, 10], 4, 0),
        ([1, 4, 8, 10, 20], 1, 18),
        ([1, 4, 8, 10, 20], 5, 0),
        ([1, 2, 3, 4, 5], 2, 4),
        ([1, 100], 1, 99),
        ([1, 100], 2, 0),
        ([5], 1, 0)
    ]
    
    print("Testing Allocate Mailboxes Solutions:")
    print("=" * 70)
    
    for i, (houses, k, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Houses: {houses}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(houses) <= 8:
            try:
                brute = min_distance_brute_force(houses[:], k)
                print(f"Brute Force:      {brute:>5} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Error")
        
        memo_result = min_distance_memoization(houses[:], k)
        dp_result = min_distance_dp_optimized(houses[:], k)
        space_result = min_distance_space_optimized(houses[:], k)
        
        print(f"Memoization:      {memo_result:>5} {'✓' if memo_result == expected else '✗'}")
        print(f"DP Optimized:     {dp_result:>5} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_result:>5} {'✓' if space_result == expected else '✗'}")
        
        # Show positions for small cases
        if len(houses) <= 8:
            min_cost, positions = min_distance_with_positions(houses[:], k)
            print(f"Positions: {positions}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    min_distance_analysis([1, 4, 8, 10, 20], 3)
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. MEDIAN OPTIMALITY: Single mailbox optimally placed at median")
    print("2. INTERVAL DP: Partition houses optimally among k mailboxes")
    print("3. PRECOMPUTATION: Calculate all single-mailbox costs first")
    print("4. SUBPROBLEM OVERLAP: Many ways to partition create same subproblems")
    print("5. GREEDY WON'T WORK: Need DP because optimal substructure is complex")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all possible house groupings")
    print("Memoization:      Top-down DP with caching")
    print("DP Optimized:     Bottom-up with precomputed costs")
    print("Space Optimized:  Rolling arrays for memory efficiency")
    print("With Positions:   DP + backtracking for actual placements")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(S(n,k) × n^k), Space: O(n)")
    print("Memoization:      Time: O(n²k),          Space: O(nk)")
    print("DP Optimized:     Time: O(n²k),          Space: O(n²)")
    print("Space Optimized:  Time: O(n²k),          Space: O(n²)")
    print("With Positions:   Time: O(n²k),          Space: O(n²)")


if __name__ == "__main__":
    test_min_distance()


"""
PATTERN RECOGNITION:
==================
Allocate Mailboxes is a classic interval partitioning optimization problem:
- Partition sorted array into k groups to minimize total cost
- Each group served optimally by single facility (mailbox at median)
- Demonstrates the power of precomputation + dynamic programming
- Foundation for many facility location and resource allocation problems

KEY INSIGHT - MEDIAN OPTIMALITY:
================================
**Single Mailbox Placement**: For any set of houses on a line, the optimal mailbox 
position that minimizes total distance is the MEDIAN position.

**Proof Intuition**: 
- Moving mailbox left/right from median increases distance to more houses than it decreases
- Median minimizes sum of absolute deviations (well-known result)
- For even number of houses, any position between two middle houses is optimal

**Mathematical Foundation**:
```
For houses at positions h₁ ≤ h₂ ≤ ... ≤ hₙ
Optimal mailbox position = h₍ₙ₊₁₎/₂₎ (median)
Minimum cost = Σᵢ |hᵢ - median|
```

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(S(n,k) × n^k) time - Stirling numbers explosion
   - Try all ways to partition n houses into k groups
   - Exponential in both n and k
   - Only viable for tiny inputs

2. **Memoization**: O(n²k) time, O(nk) space
   - State: (house_index, mailboxes_remaining)
   - Transition: try all possible group endings
   - Natural recursive structure

3. **Bottom-up DP**: O(n²k) time, O(n²) space
   - Precompute all single-mailbox costs: O(n²) time/space
   - DP[i][m] = min cost for first i houses with m mailboxes
   - Optimal time complexity for this problem

4. **Space Optimized**: O(n²k) time, O(n²) space
   - Use rolling arrays for DP computation
   - Cost matrix still needed for precomputed values
   - Minimal space usage while maintaining efficiency

DP STATE DEFINITION:
===================
```
dp[i][m] = minimum cost to allocate m mailboxes for first i houses
```

**Base Cases**:
- dp[0][0] = 0 (no houses, no mailboxes)
- dp[i][0] = ∞ for i > 0 (houses without mailboxes)
- dp[0][m] = ∞ for m > 0 (mailboxes without houses)

**Recurrence Relation**:
```
dp[i][m] = min(dp[j][m-1] + cost[j][i-1]) for all valid j

where cost[j][i-1] = optimal cost for houses[j:i] with one mailbox
```

PRECOMPUTATION OPTIMIZATION:
===========================
**Key Insight**: Since we need costs for ALL possible house segments, 
precompute them once in O(n²) time.

```python
cost[i][j] = minimum cost to serve houses[i:j+1] with one mailbox

for i in range(n):
    for j in range(i, n):
        median_pos = (i + j) // 2
        mailbox_position = houses[median_pos]
        cost[i][j] = sum(abs(houses[k] - mailbox_position) for k in range(i, j+1))
```

**Alternative Efficient Computation**:
```python
# For segment houses[i:j+1], cost can be computed incrementally
for length in range(1, n+1):
    for i in range(n - length + 1):
        j = i + length - 1
        median = houses[(i + j) // 2]
        cost[i][j] = sum(abs(houses[k] - median) for k in range(i, j+1))
```

TRANSITION ANALYSIS:
===================
**State Transition**: For dp[i][m], consider all ways to place the m-th mailbox

```
Last mailbox serves houses[j:i-1] for some j < i
Previous (m-1) mailboxes serve houses[0:j-1]

dp[i][m] = min over j of (dp[j][m-1] + cost[j][i-1])

Constraints:
- j ≥ m-1 (need at least m-1 houses for m-1 mailboxes)
- j < i (last mailbox must serve at least one house)
```

SPACE OPTIMIZATION TECHNIQUES:
=============================
**Rolling Array Optimization**:
```python
# Instead of full dp[n+1][k+1] table
prev_dp = [inf] * (n + 1)
prev_dp[0] = 0

for m in range(1, k + 1):
    curr_dp = [inf] * (n + 1)
    for i in range(m, n + 1):
        for j in range(m-1, i):
            if prev_dp[j] != inf:
                curr_dp[i] = min(curr_dp[i], prev_dp[j] + cost[j][i-1])
    prev_dp = curr_dp
```

**Memory Layout Optimization**:
- Cost matrix: O(n²) - unavoidable for this approach
- DP computation: O(n) with rolling arrays
- Total: O(n²) space complexity

SOLUTION RECONSTRUCTION:
=======================
**Finding Actual Mailbox Positions**:
1. **Track parent pointers** during DP computation
2. **Backtrack** from dp[n][k] to reconstruct partition points
3. **For each segment**, place mailbox at median position

```python
# During DP computation
if new_cost < dp[i][m]:
    dp[i][m] = new_cost
    parent[i][m] = j  # Remember which j gave optimal cost

# Reconstruction
segments = []
i, m = n, k
while m > 0:
    j = parent[i][m]
    segments.append((j, i-1))  # Segment houses[j:i-1]
    i, m = j, m-1

# Place mailboxes at medians of segments
mailbox_positions = []
for start, end in segments:
    median_idx = (start + end) // 2
    mailbox_positions.append(houses[median_idx])
```

APPLICATIONS:
============
1. **Facility Location**: Hospitals, fire stations, schools placement
2. **Network Design**: Server placement to minimize latency
3. **Supply Chain**: Warehouse location optimization
4. **Urban Planning**: Public service facility placement
5. **Telecommunications**: Cell tower placement for coverage

VARIANTS TO PRACTICE:
====================
- K-means Clustering (similar median-based partitioning)
- Minimum Cost to Cut a Stick (1547) - similar interval partitioning
- Split Array Largest Sum (410) - minimize maximum instead of sum
- Capacity To Ship Packages (1011) - related optimization problem

INTERVIEW TIPS:
==============
1. **Recognize interval partitioning**: Key pattern for this problem class
2. **Median optimality**: Crucial insight for single facility placement
3. **Precomputation strategy**: Show how to optimize repeated calculations
4. **DP state design**: Explain why (houses, mailboxes) is natural state
5. **Space optimization**: Demonstrate rolling array technique
6. **Solution reconstruction**: How to find actual mailbox positions
7. **Edge cases**: k=1, k=n, single house, identical positions
8. **Real applications**: Facility location, network design
9. **Alternative approaches**: Why greedy doesn't work, when approximation helps
10. **Complexity trade-offs**: Time vs space optimizations

MATHEMATICAL INSIGHT:
====================
This problem demonstrates **optimal facility location theory**:
- **Median minimizes sum of absolute deviations** (classical result)
- **Dynamic programming handles interaction** between facility placement decisions
- **Precomputation transforms** O(n³k) naive approach to O(n²k) optimal

The combination of **geometric optimization** (median placement) with 
**combinatorial optimization** (optimal partitioning) creates a rich problem 
that bridges discrete and continuous optimization domains.

The problem showcases how **mathematical properties** (median optimality) can be 
leveraged to design efficient algorithms for complex optimization problems.
"""
