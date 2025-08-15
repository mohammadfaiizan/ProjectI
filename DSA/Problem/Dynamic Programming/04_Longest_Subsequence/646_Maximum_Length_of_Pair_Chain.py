"""
LeetCode 646: Maximum Length of Pair Chain
Difficulty: Medium
Category: Longest Subsequence Problems (Interval LIS variant)

PROBLEM DESCRIPTION:
===================
You are given an array of n pairs where pairs[i] = [left_i, right_i] and left_i < right_i.

A pair p2 = [c, d] follows a pair p1 = [a, b] if b < c. A chain of pairs can be formed in this fashion.

Return the length of the longest chain which can be formed.

You do not need to use up all the given pairs. You can select pairs in any order.

Example 1:
Input: pairs = [[1,2],[2,3],[3,4]]
Output: 2
Explanation: The longest chain is [1,2] -> [3,4].

Example 2:
Input: pairs = [[1,2],[7,8],[4,5]]
Output: 3

Constraints:
- n == pairs.length
- 1 <= n <= 1000
- -1000 <= left_i < right_i <= 1000
"""

def find_longest_chain_brute_force(pairs):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible chains using recursion.
    
    Time Complexity: O(2^n) - exponential subsets
    Space Complexity: O(n) - recursion stack
    """
    def can_follow(pair1, pair2):
        return pair1[1] < pair2[0]
    
    def max_chain_length(index, last_pair):
        if index >= len(pairs):
            return 0
        
        # Option 1: Skip current pair
        skip = max_chain_length(index + 1, last_pair)
        
        # Option 2: Include current pair if valid
        include = 0
        if last_pair is None or can_follow(last_pair, pairs[index]):
            include = 1 + max_chain_length(index + 1, pairs[index])
        
        return max(skip, include)
    
    return max_chain_length(0, None)


def find_longest_chain_dp_lis(pairs):
    """
    DP APPROACH - LIS STYLE:
    =======================
    Sort pairs and apply LIS-like DP.
    
    Time Complexity: O(n^2) - nested loops after sorting
    Space Complexity: O(n) - DP array
    """
    if not pairs:
        return 0
    
    # Sort by starting point (or ending point)
    pairs.sort()
    n = len(pairs)
    
    # dp[i] = maximum chain length ending at index i
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            # Check if pair j can be followed by pair i
            if pairs[j][1] < pairs[i][0]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def find_longest_chain_greedy(pairs):
    """
    GREEDY APPROACH (OPTIMAL):
    =========================
    Sort by ending points and greedily select non-overlapping pairs.
    
    Time Complexity: O(n log n) - sorting
    Space Complexity: O(1) - constant extra space
    """
    if not pairs:
        return 0
    
    # Sort by ending points
    pairs.sort(key=lambda x: x[1])
    
    count = 1
    last_end = pairs[0][1]
    
    for i in range(1, len(pairs)):
        # If current pair can follow the last selected pair
        if pairs[i][0] > last_end:
            count += 1
            last_end = pairs[i][1]
    
    return count


def find_longest_chain_memoization(pairs):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization with sorted pairs.
    
    Time Complexity: O(n^2) - memoized states
    Space Complexity: O(n^2) - memoization table
    """
    if not pairs:
        return 0
    
    pairs.sort()
    memo = {}
    
    def dp(index, last_index):
        if index >= len(pairs):
            return 0
        
        if (index, last_index) in memo:
            return memo[(index, last_index)]
        
        # Skip current pair
        skip = dp(index + 1, last_index)
        
        # Include current pair if valid
        include = 0
        if last_index == -1 or pairs[last_index][1] < pairs[index][0]:
            include = 1 + dp(index + 1, index)
        
        result = max(skip, include)
        memo[(index, last_index)] = result
        return result
    
    return dp(0, -1)


def find_longest_chain_binary_search(pairs):
    """
    BINARY SEARCH APPROACH:
    =======================
    Use binary search similar to LIS optimization.
    
    Time Complexity: O(n log n) - sorting + binary search
    Space Complexity: O(n) - tails array
    """
    if not pairs:
        return 0
    
    # Sort by starting points
    pairs.sort()
    
    # tails[i] = ending point of chain of length i+1
    tails = []
    
    for start, end in pairs:
        # Binary search for position to insert/replace
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < start:
                left = mid + 1
            else:
                right = mid
        
        # Insert or replace
        if left == len(tails):
            tails.append(end)
        else:
            tails[left] = end
    
    return len(tails)


def find_longest_chain_with_chain(pairs):
    """
    FIND ACTUAL CHAIN:
    ==================
    Return both length and one possible maximum chain.
    
    Time Complexity: O(n^2) - DP + reconstruction
    Space Complexity: O(n) - DP array + parent tracking
    """
    if not pairs:
        return 0, []
    
    pairs.sort()
    n = len(pairs)
    
    dp = [1] * n
    parent = [-1] * n
    
    max_length = 1
    max_index = 0
    
    for i in range(1, n):
        for j in range(i):
            if pairs[j][1] < pairs[i][0] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
        
        if dp[i] > max_length:
            max_length = dp[i]
            max_index = i
    
    # Reconstruct chain
    chain = []
    current = max_index
    
    while current != -1:
        chain.append(pairs[current])
        current = parent[current]
    
    chain.reverse()
    return max_length, chain


def find_longest_chain_greedy_with_chain(pairs):
    """
    GREEDY WITH ACTUAL CHAIN:
    ========================
    Use greedy approach and return actual chain.
    
    Time Complexity: O(n log n) - sorting
    Space Complexity: O(n) - store chain
    """
    if not pairs:
        return 0, []
    
    # Sort by ending points
    pairs.sort(key=lambda x: x[1])
    
    chain = [pairs[0]]
    last_end = pairs[0][1]
    
    for i in range(1, len(pairs)):
        if pairs[i][0] > last_end:
            chain.append(pairs[i])
            last_end = pairs[i][1]
    
    return len(chain), chain


def find_longest_chain_activity_selection(pairs):
    """
    ACTIVITY SELECTION APPROACH:
    ============================
    Treat as classic activity selection problem.
    
    Time Complexity: O(n log n) - sorting
    Space Complexity: O(1) - constant space
    """
    # This is essentially the activity selection problem
    # Select maximum number of non-overlapping activities
    
    if not pairs:
        return 0
    
    # Sort by finish time (ending points)
    pairs.sort(key=lambda x: x[1])
    
    count = 1
    last_finish = pairs[0][1]
    
    for i in range(1, len(pairs)):
        # Current activity's start time > last activity's finish time
        if pairs[i][0] > last_finish:
            count += 1
            last_finish = pairs[i][1]
    
    return count


def find_longest_chain_all_approaches_comparison(pairs):
    """
    COMPARE ALL APPROACHES:
    ======================
    Run all approaches and compare results.
    
    Time Complexity: Varies by approach
    Space Complexity: Varies by approach
    """
    print(f"Input pairs: {pairs}")
    
    if len(pairs) <= 10:
        brute = find_longest_chain_brute_force(pairs.copy())
        print(f"Brute Force: {brute}")
    
    dp_lis = find_longest_chain_dp_lis(pairs.copy())
    greedy = find_longest_chain_greedy(pairs.copy())
    memo = find_longest_chain_memoization(pairs.copy())
    binary = find_longest_chain_binary_search(pairs.copy())
    activity = find_longest_chain_activity_selection(pairs.copy())
    
    print(f"DP (LIS style): {dp_lis}")
    print(f"Greedy: {greedy}")
    print(f"Memoization: {memo}")
    print(f"Binary Search: {binary}")
    print(f"Activity Selection: {activity}")
    
    # Show actual chains
    length, dp_chain = find_longest_chain_with_chain(pairs.copy())
    greedy_length, greedy_chain = find_longest_chain_greedy_with_chain(pairs.copy())
    
    print(f"DP Chain: {dp_chain}")
    print(f"Greedy Chain: {greedy_chain}")
    
    return greedy  # Return greedy result (optimal)


# Test cases
def test_find_longest_chain():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[1,2],[2,3],[3,4]], 2),
        ([[1,2],[7,8],[4,5]], 3),
        ([[1,2]], 1),
        ([[1,2],[2,3]], 1),
        ([[1,3],[2,4],[3,5]], 2),
        ([[-10,-8],[8,9],[-5,0],[6,10],[-6,-4],[1,7],[9,10],[-4,7]], 4),
        ([[1,2],[3,4],[5,6],[7,8]], 4),
        ([[1,10],[2,3],[4,5],[6,7]], 3),
        ([[5,6],[1,2],[4,5],[1,3]], 3),
        ([[-6,9],[1,6],[8,10],[-1,4],[-6,-2],[-9,8],[-5,3],[0,3]], 3)
    ]
    
    print("Testing Maximum Length of Pair Chain Solutions:")
    print("=" * 70)
    
    for i, (pairs, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: pairs = {pairs}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(pairs) <= 8:
            brute = find_longest_chain_brute_force(pairs.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        dp_lis = find_longest_chain_dp_lis(pairs.copy())
        greedy = find_longest_chain_greedy(pairs.copy())
        memo = find_longest_chain_memoization(pairs.copy())
        binary = find_longest_chain_binary_search(pairs.copy())
        activity = find_longest_chain_activity_selection(pairs.copy())
        
        print(f"DP (LIS):         {dp_lis:>3} {'✓' if dp_lis == expected else '✗'}")
        print(f"Greedy:           {greedy:>3} {'✓' if greedy == expected else '✗'}")
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Binary Search:    {binary:>3} {'✓' if binary == expected else '✗'}")
        print(f"Activity Select:  {activity:>3} {'✓' if activity == expected else '✗'}")
        
        # Show actual chain for small cases
        if expected > 1 and len(pairs) <= 8:
            length, chain = find_longest_chain_greedy_with_chain(pairs.copy())
            print(f"Optimal Chain: {chain}")
    
    # Detailed comparison example
    print(f"\n" + "=" * 70)
    print("DETAILED COMPARISON EXAMPLE:")
    print("-" * 40)
    find_longest_chain_all_approaches_comparison([[1,2],[2,3],[3,4]])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. GREEDY OPTIMAL: Sort by ending points, select greedily")
    print("2. ACTIVITY SELECTION: Classic interval scheduling problem")
    print("3. LIS VARIANT: Can use LIS DP, but greedy is better")
    print("4. SORTING STRATEGY: End points for greedy, start points for LIS")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all possible chains")
    print("DP (LIS):         Sort + LIS-style DP")
    print("Greedy:           Sort by end + greedy selection")
    print("Memoization:      Recursive with caching")
    print("Binary Search:    LIS with binary search optimization")
    print("Activity Select:  Classic greedy algorithm")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),     Space: O(n)")
    print("DP (LIS):         Time: O(n²),      Space: O(n)")
    print("Greedy:           Time: O(n log n), Space: O(1)  ← OPTIMAL")
    print("Memoization:      Time: O(n²),      Space: O(n²)")
    print("Binary Search:    Time: O(n log n), Space: O(n)")
    print("Activity Select:  Time: O(n log n), Space: O(1)")


if __name__ == "__main__":
    test_find_longest_chain()


"""
PATTERN RECOGNITION:
==================
This is a classic Activity Selection problem disguised as LIS:
- Intervals/pairs instead of single values
- "Following" relationship: pair1[1] < pair2[0]
- Can be solved optimally with greedy algorithm
- Also solvable with LIS-style DP (suboptimal but educational)

KEY INSIGHT - GREEDY OPTIMALITY:
===============================
**Why greedy works**:
1. Sort pairs by ending points
2. Always select pair that ends earliest among remaining
3. This leaves maximum room for future selections

**Proof of optimality**:
- Let OPT be optimal solution, GREEDY be greedy solution
- If they differ, can always replace earliest pair in OPT with greedy choice
- This replacement is valid and doesn't reduce solution quality

ALGORITHM APPROACHES:
====================

1. **Greedy (Optimal)**: O(n log n)
   - Sort by ending points
   - Select greedily by earliest end time
   - Classic activity selection algorithm

2. **DP (LIS-style)**: O(n²)
   - Sort by starting points
   - Apply LIS logic with interval constraint
   - Educational but suboptimal

3. **Binary Search**: O(n log n)
   - LIS with binary search optimization
   - More complex than greedy for same complexity

4. **Brute Force**: O(2^n)
   - Try all possible subsets
   - Check validity of each chain

GREEDY ALGORITHM:
================
```python
pairs.sort(key=lambda x: x[1])  # Sort by ending points
count = 1
last_end = pairs[0][1]

for i in range(1, len(pairs)):
    if pairs[i][0] > last_end:  # Can follow
        count += 1
        last_end = pairs[i][1]
```

DP RECURRENCE:
=============
After sorting by starting points:
```
dp[i] = max(dp[j] + 1) for all j < i where pairs[j][1] < pairs[i][0]
```

SORTING STRATEGIES:
==================
- **For Greedy**: Sort by ending points (finish time)
- **For DP**: Sort by starting points (or ending points)
- **For Binary Search**: Sort by starting points

COMPARISON WITH LIS:
===================
- **LIS**: Single values, simple comparison
- **Pair Chain**: Intervals, non-overlapping constraint
- **Solution**: Greedy optimal for intervals, DP for general LIS

APPLICATIONS:
============
1. **Scheduling**: Activity/job scheduling
2. **Resource Allocation**: Non-overlapping resource usage
3. **Event Planning**: Maximum non-conflicting events
4. **Interval Problems**: General interval optimization
5. **Bioinformatics**: Gene sequence intervals

VARIANTS TO PRACTICE:
====================
- Activity Selection Problem - exact same problem
- Russian Doll Envelopes (354) - 2D version with strict ordering
- Non-overlapping Intervals (435) - minimum removals version
- Merge Intervals (56) - interval merging problem

EDGE CASES:
==========
1. **Single pair**: Return 1
2. **All overlapping**: Return 1
3. **No overlapping**: Return n
4. **Identical pairs**: Handle duplicates properly
5. **Negative coordinates**: Algorithm works unchanged

INTERVIEW TIPS:
==============
1. **Recognize as activity selection**: Key insight
2. **Show both approaches**: Greedy (optimal) and DP (educational)
3. **Prove greedy correctness**: Exchange argument
4. **Explain sorting choice**: Why end points for greedy
5. **Handle edge cases**: Single pair, all overlapping
6. **Discuss complexity**: Why greedy is better than DP
7. **Show actual chain**: Reconstruction algorithm
8. **Related problems**: Connect to other interval problems
9. **Real applications**: Scheduling, resource allocation
10. **Variants**: What if we need weighted intervals?

MATHEMATICAL INSIGHT:
====================
This is the classic "Interval Scheduling Maximization" problem:
- Greedy choice property: Always safe to choose earliest ending interval
- Optimal substructure: Remaining problem is same type
- Matroid structure: Independent sets form a matroid

The greedy algorithm is optimal because intervals form a special structure
where local optimal choices lead to global optimum.
"""
