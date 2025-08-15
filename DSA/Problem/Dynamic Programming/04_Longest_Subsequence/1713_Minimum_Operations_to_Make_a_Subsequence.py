"""
LeetCode 1713: Minimum Operations to Make a Subsequence
Difficulty: Hard
Category: Longest Subsequence Problems (LCS with Constraints)

PROBLEM DESCRIPTION:
===================
You are given an array target and an array arr.

In one operation, you can insert any integer at any position in arr.

Return the minimum number of operations to make target a subsequence of arr.

A subsequence of an array is a new array generated from the original array by deleting some (possibly zero) 
elements without changing the remaining elements' relative order.

Example 1:
Input: target = [5,1,3], arr = [9,4,2,3,4]
Output: 2
Explanation: You can add 5 and 1 in such a way that makes arr = [5,9,4,1,2,3,4], then target will be a subsequence of arr.

Example 2:
Input: target = [6,4,8,1,3,2], arr = [4,7,6,2,3,8,6,1]
Output: 3

Constraints:
- 1 <= target.length, arr.length <= 10^5
- 1 <= target[i], arr[i] <= 10^9
- target contains no duplicates.
"""

def min_operations_brute_force(target, arr):
    """
    BRUTE FORCE APPROACH - LCS:
    ===========================
    Find LCS between target and arr, then calculate operations needed.
    
    Time Complexity: O(n * m) - LCS computation
    Space Complexity: O(n * m) - DP table
    """
    def lcs_length(text1, text2):
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    # LCS gives us maximum elements we don't need to insert
    lcs_len = lcs_length(target, arr)
    return len(target) - lcs_len


def min_operations_optimized_lcs(target, arr):
    """
    OPTIMIZED LCS WITH COORDINATE COMPRESSION:
    =========================================
    Use the fact that target has no duplicates for optimization.
    
    Time Complexity: O(n * m) - but with optimizations
    Space Complexity: O(min(n, m)) - space-optimized LCS
    """
    # Create mapping from target elements to their indices
    target_indices = {val: i for i, val in enumerate(target)}
    
    # Filter arr to only include elements that exist in target
    filtered_arr = []
    for val in arr:
        if val in target_indices:
            filtered_arr.append(target_indices[val])
    
    if not filtered_arr:
        return len(target)
    
    # Now find LIS in filtered_arr (indices are from target)
    def lis_length(nums):
        if not nums:
            return 0
        
        from bisect import bisect_left
        tails = []
        
        for num in nums:
            pos = bisect_left(tails, num)
            if pos == len(tails):
                tails.append(num)
            else:
                tails[pos] = num
        
        return len(tails)
    
    # LIS of indices corresponds to LCS of original arrays
    lis_len = lis_length(filtered_arr)
    return len(target) - lis_len


def min_operations_lis_transformation(target, arr):
    """
    LIS TRANSFORMATION APPROACH (OPTIMAL):
    =====================================
    Transform to LIS problem using coordinate mapping.
    
    Time Complexity: O(m log m) - LIS with binary search
    Space Complexity: O(m) - LIS array
    """
    # Map target elements to their positions
    target_pos = {val: i for i, val in enumerate(target)}
    
    # Convert arr to sequence of target positions
    positions = []
    for val in arr:
        if val in target_pos:
            positions.append(target_pos[val])
    
    # Find LIS in positions - this gives us the LCS length
    def lis_binary_search(nums):
        if not nums:
            return 0
        
        import bisect
        tails = []
        
        for num in nums:
            idx = bisect.bisect_left(tails, num)
            if idx == len(tails):
                tails.append(num)
            else:
                tails[idx] = num
        
        return len(tails)
    
    max_common = lis_binary_search(positions)
    return len(target) - max_common


def min_operations_dp_with_mapping(target, arr):
    """
    DP WITH POSITION MAPPING:
    ========================
    Use DP on position mappings for clearer understanding.
    
    Time Complexity: O(m^2) - DP on positions
    Space Complexity: O(m) - DP array
    """
    # Create position mapping
    target_pos = {val: i for i, val in enumerate(target)}
    
    # Extract valid positions from arr
    positions = []
    for val in arr:
        if val in target_pos:
            positions.append(target_pos[val])
    
    if not positions:
        return len(target)
    
    # DP for LIS
    n = len(positions)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if positions[j] < positions[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    max_lis = max(dp) if dp else 0
    return len(target) - max_lis


def min_operations_segment_tree(target, arr):
    """
    SEGMENT TREE APPROACH:
    =====================
    Use segment tree for range maximum queries.
    
    Time Complexity: O(m log n) - segment tree operations
    Space Complexity: O(n) - segment tree
    """
    # Position mapping
    target_pos = {val: i for i, val in enumerate(target)}
    
    positions = []
    for val in arr:
        if val in target_pos:
            positions.append(target_pos[val])
    
    if not positions:
        return len(target)
    
    # Segment tree for LIS
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
    
    seg_tree = SegmentTree(len(target))
    max_lis = 0
    
    for pos in positions:
        # Query maximum LIS length for positions < pos
        prev_max = seg_tree.query(1, 0, len(target) - 1, 0, pos - 1) if pos > 0 else 0
        
        # Update with new LIS length
        current_lis = prev_max + 1
        seg_tree.update(1, 0, len(target) - 1, pos, current_lis)
        
        max_lis = max(max_lis, current_lis)
    
    return len(target) - max_lis


def min_operations_with_subsequence(target, arr):
    """
    FIND ACTUAL SUBSEQUENCE:
    =======================
    Return operations needed and the actual common subsequence.
    
    Time Complexity: O(m log m) - LIS with reconstruction
    Space Complexity: O(m) - subsequence storage
    """
    # Position mapping
    target_pos = {val: i for i, val in enumerate(target)}
    
    # Extract positions and original values
    positions_with_vals = []
    for val in arr:
        if val in target_pos:
            positions_with_vals.append((target_pos[val], val))
    
    if not positions_with_vals:
        return len(target), []
    
    # Find LIS with values
    import bisect
    
    def lis_with_values(pos_val_pairs):
        if not pos_val_pairs:
            return 0, []
        
        # tails[i] = (position, value) with smallest position for LIS of length i+1
        tails = []
        # parent[i] = index of previous element in LIS ending at i
        parent = [-1] * len(pos_val_pairs)
        # pos_to_idx[i] = index in pos_val_pairs for tails[i]
        pos_to_idx = []
        
        for i, (pos, val) in enumerate(pos_val_pairs):
            # Binary search for position to insert/replace
            left, right = 0, len(tails)
            while left < right:
                mid = (left + right) // 2
                if tails[mid][0] < pos:
                    left = mid + 1
                else:
                    right = mid
            
            # Update parent pointer
            if left > 0:
                parent[i] = pos_to_idx[left - 1]
            
            # Update tails
            if left == len(tails):
                tails.append((pos, val))
                pos_to_idx.append(i)
            else:
                tails[left] = (pos, val)
                pos_to_idx[left] = i
        
        # Reconstruct LIS
        if not tails:
            return 0, []
        
        lis_vals = []
        current_idx = pos_to_idx[-1]
        
        while current_idx != -1:
            lis_vals.append(pos_val_pairs[current_idx][1])
            current_idx = parent[current_idx]
        
        lis_vals.reverse()
        return len(lis_vals), lis_vals
    
    lis_len, common_subseq = lis_with_values(positions_with_vals)
    return len(target) - lis_len, common_subseq


def min_operations_analysis(target, arr):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step transformation and analysis.
    
    Time Complexity: O(m log m) - analysis + LIS
    Space Complexity: O(m) - temporary arrays
    """
    print(f"Target: {target}")
    print(f"Array:  {arr}")
    
    # Step 1: Create position mapping
    target_pos = {val: i for i, val in enumerate(target)}
    print(f"\nPosition mapping: {target_pos}")
    
    # Step 2: Transform arr to positions
    positions = []
    print(f"\nTransformation:")
    for i, val in enumerate(arr):
        if val in target_pos:
            pos = target_pos[val]
            positions.append(pos)
            print(f"  arr[{i}] = {val} -> position {pos} in target")
        else:
            print(f"  arr[{i}] = {val} -> not in target (skip)")
    
    print(f"\nPosition sequence: {positions}")
    
    # Step 3: Find LIS
    if positions:
        import bisect
        tails = []
        lis_trace = []
        
        for pos in positions:
            idx = bisect.bisect_left(tails, pos)
            if idx == len(tails):
                tails.append(pos)
            else:
                tails[idx] = pos
            lis_trace.append(f"Process {pos}: tails = {tails[:]}")
        
        print(f"\nLIS computation:")
        for trace in lis_trace:
            print(f"  {trace}")
        
        lis_length = len(tails)
        print(f"\nLIS length: {lis_length}")
        print(f"Common elements: {lis_length}")
        print(f"Operations needed: {len(target)} - {lis_length} = {len(target) - lis_length}")
    else:
        print(f"\nNo common elements found")
        print(f"Operations needed: {len(target)}")
    
    return min_operations_lis_transformation(target, arr)


# Test cases
def test_min_operations():
    """Test all implementations with various inputs"""
    test_cases = [
        ([5,1,3], [9,4,2,3,4], 2),
        ([6,4,8,1,3,2], [4,7,6,2,3,8,6,1], 3),
        ([1,2,3], [1,2,3], 0),
        ([1,2,3], [3,2,1], 2),
        ([1], [2,3,4], 1),
        ([1,2,3,4,5], [5,4,3,2,1], 4),
        ([16,7,20,11,15,13,10,14,6,8], [11,14,15,7,5,5,6,3,6,11,17,13,8,17,7,14,5,7,9,5], 6),
        ([5,1,3], [], 3),
        ([], [1,2,3], 0),
        ([1,3,5,4,7], [1,3,2,4,5,6,7,8], 1)
    ]
    
    print("Testing Minimum Operations to Make Subsequence Solutions:")
    print("=" * 70)
    
    for i, (target, arr, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: target = {target[:6]}{'...' if len(target) > 6 else ''}")
        print(f"arr = {arr[:8]}{'...' if len(arr) > 8 else ''}")
        print(f"Expected: {expected}")
        
        # Test approaches
        if len(target) <= 20 and len(arr) <= 20:
            brute = min_operations_brute_force(target.copy(), arr.copy())
            print(f"Brute Force LCS:  {brute:>3} {'✓' if brute == expected else '✗'}")
        
        optimized_lcs = min_operations_optimized_lcs(target.copy(), arr.copy())
        lis_transform = min_operations_lis_transformation(target.copy(), arr.copy())
        dp_mapping = min_operations_dp_with_mapping(target.copy(), arr.copy())
        
        print(f"Optimized LCS:    {optimized_lcs:>3} {'✓' if optimized_lcs == expected else '✗'}")
        print(f"LIS Transform:    {lis_transform:>3} {'✓' if lis_transform == expected else '✗'}")
        print(f"DP Mapping:       {dp_mapping:>3} {'✓' if dp_mapping == expected else '✗'}")
        
        if len(target) <= 50:
            seg_tree = min_operations_segment_tree(target.copy(), arr.copy())
            print(f"Segment Tree:     {seg_tree:>3} {'✓' if seg_tree == expected else '✗'}")
        
        # Show actual subsequence for small cases
        if len(target) <= 10 and len(arr) <= 15:
            ops, subseq = min_operations_with_subsequence(target.copy(), arr.copy())
            if subseq:
                print(f"Common subseq: {subseq}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    min_operations_analysis([5,1,3], [9,4,2,3,4])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. PROBLEM REDUCTION: LCS → LIS transformation")
    print("2. COORDINATE MAPPING: Map target elements to positions")
    print("3. LIS ON POSITIONS: Increasing positions = valid subsequence")
    print("4. OPTIMAL COMPLEXITY: O(m log m) using binary search LIS")
    print("5. NO DUPLICATES: Target constraint enables position mapping")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force LCS:  Standard LCS computation")
    print("Optimized LCS:    LCS with coordinate compression")
    print("LIS Transform:    Convert to LIS problem")
    print("DP Mapping:       DP on position mappings")
    print("Segment Tree:     Advanced range query approach")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force LCS:  Time: O(n*m),     Space: O(n*m)")
    print("Optimized LCS:    Time: O(n*m),     Space: O(min(n,m))")
    print("LIS Transform:    Time: O(m log m), Space: O(m)  ← OPTIMAL")
    print("DP Mapping:       Time: O(m²),      Space: O(m)")
    print("Segment Tree:     Time: O(m log n), Space: O(n)")


if __name__ == "__main__":
    test_min_operations()


"""
PATTERN RECOGNITION:
==================
This is a clever LCS → LIS transformation problem:
- Classic LCS would be O(n×m) which is too slow for large inputs
- Key insight: Transform to LIS problem using coordinate mapping
- Exploit constraint: target has no duplicates
- Achieves optimal O(m log m) complexity

KEY INSIGHT - LCS TO LIS TRANSFORMATION:
======================================
**Problem**: Find LCS between target and arr

**Transformation**:
1. Map each target element to its position: target[i] → i
2. Convert arr to sequence of target positions (skip non-target elements)  
3. Find LIS in the position sequence
4. LIS length = LCS length

**Why this works**:
- LCS preserves relative order in both sequences
- Position mapping preserves target order
- LIS in positions = valid subsequence maintaining target order

MATHEMATICAL PROOF:
==================
**Claim**: LIS of position sequence = LCS of original sequences

**Proof**:
- Let LIS be positions p₁ < p₂ < ... < pₖ
- This corresponds to target elements target[p₁], target[p₂], ..., target[pₖ]
- These elements appear in arr in the same relative order
- Therefore, they form a common subsequence of length k
- Conversely, any common subsequence gives an increasing position sequence

ALGORITHM APPROACHES:
====================

1. **LIS Transformation (Optimal)**: O(m log m)
   - Map target elements to positions
   - Convert arr to position sequence
   - Apply binary search LIS

2. **Optimized LCS**: O(n×m) with optimizations
   - Standard LCS with coordinate compression
   - Space optimization to O(min(n,m))

3. **Segment Tree**: O(m log n)
   - Use segment tree for LIS computation
   - Good for online/streaming versions

4. **Brute Force LCS**: O(n×m)
   - Standard LCS dynamic programming
   - Too slow for large inputs

COORDINATE MAPPING DETAILS:
==========================
```python
# Step 1: Create position mapping
target_pos = {val: i for i, val in enumerate(target)}

# Step 2: Transform arr
positions = []
for val in arr:
    if val in target_pos:
        positions.append(target_pos[val])

# Step 3: LIS on positions
lis_length = longest_increasing_subsequence(positions)

# Step 4: Calculate operations
return len(target) - lis_length
```

EXAMPLE WALKTHROUGH:
===================
```
target = [5,1,3], arr = [9,4,2,3,4]

Step 1: Position mapping
5 → 0, 1 → 1, 3 → 2

Step 2: Transform arr
[9,4,2,3,4] → [2] (only 3 appears in target at position 2)

Step 3: LIS of [2]
LIS length = 1

Step 4: Operations
3 - 1 = 2 operations needed
```

COMPLEXITY OPTIMIZATION:
========================
**Why O(m log m) is optimal**:
- Must process all m elements in arr: Ω(m)
- LIS with binary search: O(m log m)
- Position mapping: O(m) with hashmap
- Total: O(m log m)

**Space optimization**: O(m) for LIS array

EDGE CASES:
==========
1. **No common elements**: Return len(target)
2. **Empty arrays**: Handle appropriately
3. **All elements common**: Return 0
4. **Target longer than arr**: May need all target elements

APPLICATIONS:
============
1. **Sequence Alignment**: Bioinformatics applications
2. **String Matching**: Subsequence matching with insertions
3. **Data Synchronization**: Minimal operations to sync sequences
4. **Version Control**: File difference computation

VARIANTS TO PRACTICE:
====================
- Longest Common Subsequence (1143) - foundation problem
- Edit Distance (72) - more general transformation
- Is Subsequence (392) - simpler validation version
- Number of Matching Subsequences (792) - counting variant

INTERVIEW TIPS:
==============
1. **Recognize optimization opportunity**: LCS too slow for constraints
2. **Explain transformation**: Why LCS → LIS works
3. **Show coordinate mapping**: Key technique
4. **Prove correctness**: Why position ordering preserves validity
5. **Complexity analysis**: From O(n×m) to O(m log m)
6. **Handle edge cases**: Empty arrays, no common elements
7. **Alternative approaches**: Segment tree, coordinate compression
8. **Practical considerations**: Memory usage, implementation details
9. **Related problems**: Connect to LCS, LIS, edit distance
10. **Real applications**: Bioinformatics, version control systems

MATHEMATICAL INSIGHT:
====================
This problem beautifully demonstrates:
- **Problem transformation**: Converting one problem type to another
- **Constraint exploitation**: Using "no duplicates" for optimization
- **Algorithmic reduction**: LCS → LIS with different complexity

The key insight is recognizing when problem constraints enable more efficient approaches than the obvious solution.
"""
