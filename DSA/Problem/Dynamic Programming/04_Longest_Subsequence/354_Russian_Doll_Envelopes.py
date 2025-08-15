"""
LeetCode 354: Russian Doll Envelopes
Difficulty: Hard
Category: Longest Subsequence Problems (2D LIS variant)

PROBLEM DESCRIPTION:
===================
You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] represents the width 
and the height of an envelope.

One envelope can fit into another if and only if both the width and height of one envelope are 
greater than the other envelope's width and height.

Return the maximum number of envelopes you can Russian doll (i.e., put one inside another).

Note: You cannot rotate an envelope.

Example 1:
Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).

Example 2:
Input: envelopes = [[1,1],[1,1],[1,1]]
Output: 1

Constraints:
- 1 <= envelopes.length <= 10^5
- envelopes[i].length == 2
- 1 <= wi, hi <= 10^5
"""

def max_envelopes_brute_force(envelopes):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible sequences of envelopes.
    
    Time Complexity: O(2^n) - exponential subsequences
    Space Complexity: O(n) - recursion stack
    """
    def can_fit(env1, env2):
        return env1[0] < env2[0] and env1[1] < env2[1]
    
    def max_dolls(index, prev_envelope):
        if index >= len(envelopes):
            return 0
        
        # Option 1: Include current envelope if it fits
        include = 0
        if prev_envelope is None or can_fit(prev_envelope, envelopes[index]):
            include = 1 + max_dolls(index + 1, envelopes[index])
        
        # Option 2: Skip current envelope
        skip = max_dolls(index + 1, prev_envelope)
        
        return max(include, skip)
    
    return max_dolls(0, None)


def max_envelopes_dp_quadratic(envelopes):
    """
    DP APPROACH O(n^2):
    ==================
    Sort envelopes and apply LIS logic.
    
    Time Complexity: O(n^2) - DP after sorting
    Space Complexity: O(n) - DP array
    """
    if not envelopes:
        return 0
    
    # Sort by width, then by height
    envelopes.sort()
    n = len(envelopes)
    
    dp = [1] * n  # dp[i] = max envelopes ending at index i
    
    for i in range(1, n):
        for j in range(i):
            # Check if envelope j can fit into envelope i
            if (envelopes[j][0] < envelopes[i][0] and 
                envelopes[j][1] < envelopes[i][1]):
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def max_envelopes_optimal(envelopes):
    """
    OPTIMAL APPROACH - SORT + LIS:
    =============================
    Sort by width ascending, height descending, then find LIS on heights.
    
    Time Complexity: O(n log n) - sorting + LIS with binary search
    Space Complexity: O(n) - LIS array
    """
    if not envelopes:
        return 0
    
    # Key insight: Sort by width ascending, height descending
    # This ensures that for same width, we can't pick multiple envelopes
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    
    # Extract heights and find LIS
    heights = [env[1] for env in envelopes]
    
    return length_of_lis_binary_search(heights)


def length_of_lis_binary_search(nums):
    """
    Binary search LIS implementation.
    
    Time Complexity: O(n log n) - binary search for each element
    Space Complexity: O(n) - tails array
    """
    if not nums:
        return 0
    
    tails = []
    
    for num in nums:
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)


def max_envelopes_with_sequence(envelopes):
    """
    FIND ACTUAL SEQUENCE:
    ====================
    Return both count and actual sequence of envelopes.
    
    Time Complexity: O(n^2) - DP + reconstruction
    Space Complexity: O(n) - DP array + parent tracking
    """
    if not envelopes:
        return 0, []
    
    # Sort envelopes
    envelopes.sort()
    n = len(envelopes)
    
    dp = [1] * n
    parent = [-1] * n
    
    max_length = 1
    max_index = 0
    
    for i in range(1, n):
        for j in range(i):
            if (envelopes[j][0] < envelopes[i][0] and 
                envelopes[j][1] < envelopes[i][1]):
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        
        if dp[i] > max_length:
            max_length = dp[i]
            max_index = i
    
    # Reconstruct sequence
    sequence = []
    current = max_index
    
    while current != -1:
        sequence.append(envelopes[current])
        current = parent[current]
    
    sequence.reverse()
    return max_length, sequence


def max_envelopes_coordinate_compression(envelopes):
    """
    COORDINATE COMPRESSION APPROACH:
    ===============================
    Use coordinate compression with segment tree for better performance.
    
    Time Complexity: O(n log n) - coordinate compression + segment tree
    Space Complexity: O(n) - segment tree
    """
    if not envelopes:
        return 0
    
    # Sort by width ascending, height descending
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    
    # Extract unique heights for coordinate compression
    heights = [env[1] for env in envelopes]
    unique_heights = sorted(set(heights))
    height_to_index = {h: i for i, h in enumerate(unique_heights)}
    
    # Segment tree for range maximum queries
    class SegmentTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (4 * size)
        
        def update(self, node, start, end, idx, val):
            if start == end:
                self.tree[node] = max(self.tree[node], val)
            else:
                mid = (start + end) // 2
                if idx <= mid:
                    self.update(2 * node, start, mid, idx, val)
                else:
                    self.update(2 * node + 1, mid + 1, end, idx, val)
                self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
        
        def query(self, node, start, end, l, r):
            if r < start or end < l:
                return 0
            if l <= start and end <= r:
                return self.tree[node]
            
            mid = (start + end) // 2
            left_max = self.query(2 * node, start, mid, l, r)
            right_max = self.query(2 * node + 1, mid + 1, end, l, r)
            return max(left_max, right_max)
    
    seg_tree = SegmentTree(len(unique_heights))
    max_envelopes = 0
    
    for width, height in envelopes:
        height_idx = height_to_index[height]
        
        # Query for maximum envelopes with height < current height
        current_max = seg_tree.query(1, 0, len(unique_heights) - 1, 0, height_idx - 1)
        new_count = current_max + 1
        
        # Update segment tree
        seg_tree.update(1, 0, len(unique_heights) - 1, height_idx, new_count)
        max_envelopes = max(max_envelopes, new_count)
    
    return max_envelopes


def max_envelopes_memoization(envelopes):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization with sorted envelopes.
    
    Time Complexity: O(n^2) - memoized states
    Space Complexity: O(n^2) - memoization table
    """
    if not envelopes:
        return 0
    
    envelopes.sort()
    memo = {}
    
    def dp(index, prev_index):
        if index >= len(envelopes):
            return 0
        
        if (index, prev_index) in memo:
            return memo[(index, prev_index)]
        
        # Option 1: Include current envelope if it fits
        include = 0
        if (prev_index == -1 or 
            (envelopes[prev_index][0] < envelopes[index][0] and 
             envelopes[prev_index][1] < envelopes[index][1])):
            include = 1 + dp(index + 1, index)
        
        # Option 2: Skip current envelope
        skip = dp(index + 1, prev_index)
        
        result = max(include, skip)
        memo[(index, prev_index)] = result
        return result
    
    return dp(0, -1)


def max_envelopes_detailed_analysis(envelopes):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step process and analysis.
    
    Time Complexity: O(n log n) - optimal approach
    Space Complexity: O(n) - LIS array
    """
    if not envelopes:
        return 0
    
    print(f"Original envelopes: {envelopes}")
    
    # Sort by width ascending, height descending
    envelopes.sort(key=lambda x: (x[0], -x[1]))
    print(f"Sorted envelopes: {envelopes}")
    
    # Extract heights
    heights = [env[1] for env in envelopes]
    print(f"Heights sequence: {heights}")
    
    # Apply LIS on heights
    print(f"Finding LIS on heights...")
    
    tails = []
    lis_trace = []
    
    for i, height in enumerate(heights):
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < height:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(height)
        else:
            tails[left] = height
        
        lis_trace.append(f"Height {height}: tails = {tails[:]}")
    
    for trace in lis_trace:
        print(f"  {trace}")
    
    result = len(tails)
    print(f"Maximum envelopes: {result}")
    
    return result


# Test cases
def test_max_envelopes():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[5,4],[6,4],[6,7],[2,3]], 3),
        ([[1,1],[1,1],[1,1]], 1),
        ([[1,2],[2,3],[3,4],[4,5]], 4),
        ([[4,5],[4,6],[6,7],[2,3],[1,1]], 4),
        ([[1,1]], 1),
        ([[2,1],[3,2],[4,3],[5,4]], 4),
        ([[10,8],[1,12],[6,15],[2,18]], 2),
        ([[1,3],[3,5],[6,7],[6,8],[8,4],[9,5]], 3),
        ([[15,8],[2,20],[2,14],[4,17],[8,19],[8,9],[5,7],[11,19],[8,11],[13,11],[2,13],[11,19],[8,11],[13,11],[2,13],[11,19],[16,1],[18,13],[14,17],[18,19]], 5)
    ]
    
    print("Testing Russian Doll Envelopes Solutions:")
    print("=" * 70)
    
    for i, (envelopes, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: envelopes = {envelopes}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(envelopes) <= 8:
            brute = max_envelopes_brute_force(envelopes.copy())
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        if len(envelopes) <= 15:
            dp_quad = max_envelopes_dp_quadratic(envelopes.copy())
            memo = max_envelopes_memoization(envelopes.copy())
            print(f"DP O(n²):         {dp_quad:>3} {'✓' if dp_quad == expected else '✗'}")
            print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        
        optimal = max_envelopes_optimal(envelopes.copy())
        coord_comp = max_envelopes_coordinate_compression(envelopes.copy())
        
        print(f"Optimal:          {optimal:>3} {'✓' if optimal == expected else '✗'}")
        print(f"Coord Compress:   {coord_comp:>3} {'✓' if coord_comp == expected else '✗'}")
        
        # Show actual sequence for small cases
        if expected > 1 and len(envelopes) <= 8:
            count, sequence = max_envelopes_with_sequence(envelopes.copy())
            print(f"Sequence: {sequence}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    max_envelopes_detailed_analysis([[5,4],[6,4],[6,7],[2,3]])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. SORTING STRATEGY: Width ascending, height descending")
    print("2. WHY HEIGHT DESCENDING: Prevents multiple envelopes with same width")
    print("3. REDUCTION TO LIS: After sorting, find LIS on heights")
    print("4. OPTIMAL COMPLEXITY: O(n log n) using binary search LIS")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all subsequences")
    print("DP O(n²):         Sort + classic LIS DP")
    print("Optimal:          Sort + binary search LIS")
    print("Coordinate Comp:  Advanced segment tree approach")
    print("Memoization:      Cache recursive results")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),     Space: O(n)")
    print("DP O(n²):         Time: O(n²),      Space: O(n)")
    print("Optimal:          Time: O(n log n), Space: O(n)  ← BEST")
    print("Coordinate Comp:  Time: O(n log n), Space: O(n)")
    print("Memoization:      Time: O(n²),      Space: O(n²)")


if __name__ == "__main__":
    test_max_envelopes()


"""
PATTERN RECOGNITION:
==================
This is a 2D version of Longest Increasing Subsequence:
- Instead of single values, we have 2D envelopes (width, height)
- One envelope fits in another if BOTH dimensions are strictly smaller
- Classic reduction: Sort + transform to 1D LIS problem

KEY INSIGHT - SORTING STRATEGY:
==============================
**Critical sorting approach**:
1. Sort by width ASCENDING
2. Sort by height DESCENDING (for same width)

**Why height descending?**
- Ensures we can't pick multiple envelopes with same width
- For same width, only the one with largest height can be in LIS
- This transforms 2D problem into 1D LIS on heights

MATHEMATICAL PROOF:
==================
After sorting by (width↑, height↓):
- If envelope A comes before B in sorted order, then A.width ≤ B.width
- If A and B have same width, then A.height ≥ B.height
- Therefore, A cannot fit inside B if they have same width
- This allows us to treat it as LIS problem on heights only

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(2^n)
   - Try all possible subsequences
   - Check if each forms valid Russian doll

2. **DP O(n²)**: O(n² + n log n)
   - Sort envelopes by both dimensions
   - Apply classic LIS DP logic

3. **Optimal**: O(n log n)
   - Sort by (width↑, height↓)
   - Apply binary search LIS on heights
   - Best possible complexity

4. **Coordinate Compression**: O(n log n)
   - Advanced approach using segment trees
   - Good for variants with additional constraints

SORTING EXAMPLES:
================
```
Original: [[5,4],[6,4],[6,7],[2,3]]
Sorted:   [[2,3],[5,4],[6,7],[6,4]]  // (width↑, height↓)
Heights:  [3,4,7,4]
LIS on heights: [3,4,7] → length 3
```

**Why not [6,4] after [6,7]?**
Because height 4 < 7, so it can't extend the LIS.

EDGE CASES:
==========
1. **All same size**: Return 1
2. **Single envelope**: Return 1  
3. **Strictly increasing**: Return n
4. **Strictly decreasing**: Return 1
5. **Same width different heights**: Only one can be selected

COMPARISON WITH LIS:
===================
- **LIS**: 1D values, simple comparison
- **Russian Doll**: 2D values, need BOTH dimensions strictly smaller
- **Solution**: Reduce 2D → 1D using clever sorting

STATE DEFINITION (DP):
=====================
dp[i] = maximum envelopes ending at index i (after sorting)

RECURRENCE RELATION:
===================
dp[i] = max(dp[j] + 1) for all j < i where envelope[j] fits in envelope[i]

VARIANTS TO PRACTICE:
====================
- Maximum Length of Pair Chain (646) - similar 2D problem
- Largest Divisible Subset (368) - divisibility constraint
- Box Stacking - 3D version
- Activity Selection - interval scheduling

INTERVIEW TIPS:
==============
1. **Recognize as 2D LIS**: Key insight
2. **Explain sorting strategy**: Why (width↑, height↓)
3. **Show reduction**: How 2D becomes 1D LIS
4. **Prove correctness**: Why sorting works
5. **Optimize**: From O(n²) DP to O(n log n) binary search
6. **Handle edge cases**: Same dimensions, single envelope
7. **Follow-up**: What if we need actual sequence?
8. **Generalize**: How to extend to 3D (box stacking)?
9. **Alternative**: Coordinate compression approach
10. **Complexity**: Explain why O(n log n) is optimal
"""
