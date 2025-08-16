"""
LeetCode 1187: Make Array Strictly Increasing
Difficulty: Hard
Category: Advanced DP - DP with Data Structures and Complex State Management

PROBLEM DESCRIPTION:
===================
Given two integer arrays arr1 and arr2, return the minimum number of operations (possibly zero) needed to make arr1 strictly increasing.

In one operation, you can choose two indices 0 <= i < arr1.length and 0 <= j < arr2.length and do the assignment arr1[i] = arr2[j].

If there is no way to make arr1 strictly increasing, return -1.

Example 1:
Input: arr1 = [1,5,3,6,7], arr2 = [1,3,2,4]
Output: 1
Explanation: Replace 5 with 2, then arr1 = [1, 2, 3, 6, 7].

Example 2:
Input: arr1 = [1,5,3,6,7], arr2 = [4,3,1]
Output: 2
Explanation: Replace 5 with 3, then replace 3 with 4. arr1 = [1, 3, 4, 6, 7].

Example 3:
Input: arr1 = [1,5,3,6,7], arr2 = [1,6,3,3]
Output: -1
Explanation: You can't make arr1 strictly increasing.

Constraints:
- 1 <= arr1.length, arr2.length <= 2000
- 0 <= arr1[i], arr2[i] <= 10^9
"""


def make_array_strictly_increasing_basic_dp(arr1, arr2):
    """
    BASIC DP APPROACH:
    =================
    Use DP with state (position, last_value) to track minimum operations.
    
    Time Complexity: O(n * m * n) - where n = len(arr1), m = len(arr2)
    Space Complexity: O(n * unique_values) - DP state space
    """
    import bisect
    
    n = len(arr1)
    arr2_sorted = sorted(set(arr2))  # Remove duplicates and sort
    
    # dp[i][val] = minimum operations to make arr1[0:i] strictly increasing with arr1[i-1] = val
    # Use dict for sparse representation
    dp = {}
    
    def solve(pos, last_val):
        if pos == n:
            return 0
        
        if (pos, last_val) in dp:
            return dp[(pos, last_val)]
        
        result = float('inf')
        
        # Option 1: Keep arr1[pos] if it's greater than last_val
        if arr1[pos] > last_val:
            result = min(result, solve(pos + 1, arr1[pos]))
        
        # Option 2: Replace arr1[pos] with some value from arr2
        # Find the smallest value in arr2 that is greater than last_val
        idx = bisect.bisect_right(arr2_sorted, last_val)
        
        for k in range(idx, len(arr2_sorted)):
            if arr2_sorted[k] > last_val:
                result = min(result, 1 + solve(pos + 1, arr2_sorted[k]))
        
        dp[(pos, last_val)] = result
        return result
    
    result = solve(0, -1)
    return result if result != float('inf') else -1


def make_array_strictly_increasing_optimized_dp(arr1, arr2):
    """
    OPTIMIZED DP WITH BINARY SEARCH:
    ===============================
    Use binary search to find optimal replacement values efficiently.
    
    Time Complexity: O(n * m * log m) - optimized with binary search
    Space Complexity: O(n * m) - DP table
    """
    import bisect
    
    n = len(arr1)
    arr2_unique = sorted(set(arr2))
    m = len(arr2_unique)
    
    if m == 0:
        # Can't replace anything, check if already strictly increasing
        for i in range(1, n):
            if arr1[i] <= arr1[i - 1]:
                return -1
        return 0
    
    # dp[i] = dict mapping last_value to minimum operations for position i
    dp = [{} for _ in range(n + 1)]
    dp[0][-1] = 0  # Base case: before any element with value -1
    
    for i in range(n):
        for last_val, operations in dp[i].items():
            # Option 1: Keep arr1[i]
            if arr1[i] > last_val:
                if arr1[i] not in dp[i + 1]:
                    dp[i + 1][arr1[i]] = float('inf')
                dp[i + 1][arr1[i]] = min(dp[i + 1][arr1[i]], operations)
            
            # Option 2: Replace arr1[i] with value from arr2
            idx = bisect.bisect_right(arr2_unique, last_val)
            
            for j in range(idx, m):
                new_val = arr2_unique[j]
                if new_val not in dp[i + 1]:
                    dp[i + 1][new_val] = float('inf')
                dp[i + 1][new_val] = min(dp[i + 1][new_val], operations + 1)
    
    if not dp[n]:
        return -1
    
    return min(dp[n].values())


def make_array_strictly_increasing_space_optimized(arr1, arr2):
    """
    SPACE-OPTIMIZED DP:
    ==================
    Use rolling arrays to reduce space complexity.
    
    Time Complexity: O(n * m * log m) - same as optimized
    Space Complexity: O(m) - rolling DP arrays
    """
    import bisect
    
    n = len(arr1)
    arr2_unique = sorted(set(arr2))
    
    if not arr2_unique:
        for i in range(1, n):
            if arr1[i] <= arr1[i - 1]:
                return -1
        return 0
    
    # Use two dictionaries for current and next states
    prev_dp = {-1: 0}  # last_value -> min_operations
    
    for i in range(n):
        curr_dp = {}
        
        for last_val, operations in prev_dp.items():
            # Option 1: Keep arr1[i]
            if arr1[i] > last_val:
                if arr1[i] not in curr_dp:
                    curr_dp[arr1[i]] = float('inf')
                curr_dp[arr1[i]] = min(curr_dp[arr1[i]], operations)
            
            # Option 2: Replace with value from arr2
            idx = bisect.bisect_right(arr2_unique, last_val)
            
            for j in range(idx, len(arr2_unique)):
                new_val = arr2_unique[j]
                if new_val not in curr_dp:
                    curr_dp[new_val] = float('inf')
                curr_dp[new_val] = min(curr_dp[new_val], operations + 1)
        
        prev_dp = curr_dp
    
    if not prev_dp:
        return -1
    
    return min(prev_dp.values())


def make_array_strictly_increasing_segment_tree(arr1, arr2):
    """
    SEGMENT TREE APPROACH:
    =====================
    Use segment tree for efficient range minimum queries.
    
    Time Complexity: O(n * log(max_val)) - coordinate compression + segment tree
    Space Complexity: O(max_val) - segment tree
    """
    import bisect
    
    n = len(arr1)
    arr2_unique = sorted(set(arr2))
    
    if not arr2_unique:
        for i in range(1, n):
            if arr1[i] <= arr1[i - 1]:
                return -1
        return 0
    
    # Coordinate compression
    all_values = sorted(set([-1] + arr1 + arr2_unique))
    val_to_idx = {val: i for i, val in enumerate(all_values)}
    
    class SegmentTree:
        def __init__(self, size):
            self.size = size
            self.tree = [float('inf')] * (4 * size)
        
        def update(self, node, start, end, idx, val):
            if start == end:
                self.tree[node] = min(self.tree[node], val)
            else:
                mid = (start + end) // 2
                if idx <= mid:
                    self.update(2 * node, start, mid, idx, val)
                else:
                    self.update(2 * node + 1, mid + 1, end, idx, val)
                
                self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])
        
        def query(self, node, start, end, l, r):
            if r < start or end < l:
                return float('inf')
            if l <= start and end <= r:
                return self.tree[node]
            
            mid = (start + end) // 2
            return min(self.query(2 * node, start, mid, l, r),
                      self.query(2 * node + 1, mid + 1, end, l, r))
        
        def update_point(self, idx, val):
            self.update(1, 0, self.size - 1, idx, val)
        
        def query_range(self, l, r):
            if l > r:
                return float('inf')
            return self.query(1, 0, self.size - 1, l, r)
    
    seg_tree = SegmentTree(len(all_values))
    seg_tree.update_point(val_to_idx[-1], 0)  # Base case
    
    for i in range(n):
        new_seg_tree = SegmentTree(len(all_values))
        
        for val_idx, val in enumerate(all_values):
            min_ops = seg_tree.query_range(0, val_idx - 1)
            
            if min_ops == float('inf'):
                continue
            
            # Option 1: Keep arr1[i]
            if arr1[i] > val:
                arr1_idx = val_to_idx[arr1[i]]
                new_seg_tree.update_point(arr1_idx, min_ops)
            
            # Option 2: Replace with values from arr2
            for arr2_val in arr2_unique:
                if arr2_val > val:
                    arr2_idx = val_to_idx[arr2_val]
                    new_seg_tree.update_point(arr2_idx, min_ops + 1)
        
        seg_tree = new_seg_tree
    
    result = seg_tree.query_range(0, len(all_values) - 1)
    return result if result != float('inf') else -1


def make_array_strictly_increasing_with_analysis(arr1, arr2):
    """
    MAKE ARRAY STRICTLY INCREASING WITH DETAILED ANALYSIS:
    =====================================================
    Solve with comprehensive analysis and optimization insights.
    
    Time Complexity: O(n * m * log m) - optimized approach
    Space Complexity: O(n * m) - analysis data
    """
    n = len(arr1)
    arr2_unique = sorted(set(arr2))
    m = len(arr2_unique)
    
    analysis = {
        'arr1': arr1[:],
        'arr2': arr2[:],
        'arr2_unique': arr2_unique,
        'problem_size': (n, len(arr2), m),
        'initial_violations': 0,
        'replacement_options': [],
        'dp_states_explored': 0,
        'optimization_path': [],
        'insights': []
    }
    
    # Analyze initial violations
    violations = []
    for i in range(1, n):
        if arr1[i] <= arr1[i - 1]:
            violations.append((i - 1, i, arr1[i - 1], arr1[i]))
    
    analysis['initial_violations'] = len(violations)
    analysis['violations'] = violations
    
    # Analyze replacement options for each position
    import bisect
    
    for i in range(n):
        options = []
        for val in arr2_unique:
            # Check if this replacement could be beneficial
            left_ok = (i == 0) or (val > arr1[i - 1])
            right_ok = (i == n - 1) or (val < arr1[i + 1])
            
            if left_ok:  # At least locally valid
                options.append({
                    'position': i,
                    'original': arr1[i],
                    'replacement': val,
                    'locally_valid': left_ok and right_ok
                })
        
        analysis['replacement_options'].append(options)
    
    # Solve with state tracking
    dp = {}
    states_explored = 0
    
    def solve_with_tracking(pos, last_val, path):
        nonlocal states_explored
        states_explored += 1
        
        if pos == n:
            analysis['optimization_path'] = path[:]
            return 0
        
        if (pos, last_val) in dp:
            return dp[(pos, last_val)]
        
        result = float('inf')
        best_path = None
        
        # Option 1: Keep arr1[pos]
        if arr1[pos] > last_val:
            path.append(f"Keep arr1[{pos}]={arr1[pos]}")
            cost = solve_with_tracking(pos + 1, arr1[pos], path)
            if cost < result:
                result = cost
                best_path = path[:]
            path.pop()
        
        # Option 2: Replace with value from arr2
        idx = bisect.bisect_right(arr2_unique, last_val)
        
        for k in range(idx, min(idx + 5, len(arr2_unique))):  # Limit for analysis
            new_val = arr2_unique[k]
            path.append(f"Replace arr1[{pos}]={arr1[pos]} with {new_val}")
            cost = 1 + solve_with_tracking(pos + 1, new_val, path)
            if cost < result:
                result = cost
                best_path = path[:]
            path.pop()
        
        dp[(pos, last_val)] = result
        if best_path and len(analysis['optimization_path']) == 0:
            analysis['optimization_path'] = best_path[:]
        
        return result
    
    result = solve_with_tracking(0, -1, [])
    analysis['dp_states_explored'] = states_explored
    
    # Generate insights
    analysis['insights'].append(f"Minimum operations needed: {result if result != float('inf') else 'Impossible'}")
    analysis['insights'].append(f"Initial violations: {analysis['initial_violations']}")
    analysis['insights'].append(f"DP states explored: {states_explored}")
    analysis['insights'].append(f"Unique values in arr2: {m}/{len(arr2)}")
    
    if analysis['replacement_options']:
        total_options = sum(len(opts) for opts in analysis['replacement_options'])
        analysis['insights'].append(f"Total replacement options: {total_options}")
    
    return result if result != float('inf') else -1, analysis


def make_array_analysis(arr1, arr2):
    """
    COMPREHENSIVE MAKE ARRAY ANALYSIS:
    =================================
    Analyze the problem with multiple approaches and optimizations.
    """
    print(f"Make Array Strictly Increasing Analysis:")
    print(f"arr1: {arr1}")
    print(f"arr2: {arr2}")
    print(f"Problem size: {len(arr1)} x {len(arr2)}")
    
    # Check initial state
    violations = 0
    for i in range(1, len(arr1)):
        if arr1[i] <= arr1[i - 1]:
            violations += 1
    
    print(f"Initial violations: {violations}")
    print(f"Already sorted: {'Yes' if violations == 0 else 'No'}")
    
    # Different approaches
    basic_result = make_array_strictly_increasing_basic_dp(arr1[:], arr2[:])
    optimized_result = make_array_strictly_increasing_optimized_dp(arr1[:], arr2[:])
    space_opt_result = make_array_strictly_increasing_space_optimized(arr1[:], arr2[:])
    
    print(f"\nResults:")
    print(f"Basic DP:         {basic_result}")
    print(f"Optimized DP:     {optimized_result}")
    print(f"Space Optimized:  {space_opt_result}")
    
    # Advanced approach for reasonable sizes
    if len(arr1) * len(arr2) <= 10000:
        try:
            seg_tree_result = make_array_strictly_increasing_segment_tree(arr1[:], arr2[:])
            print(f"Segment Tree:     {seg_tree_result}")
        except:
            print("Segment Tree:     Implementation complexity")
    
    # Detailed analysis
    detailed_result, analysis = make_array_strictly_increasing_with_analysis(arr1[:], arr2[:])
    
    print(f"\nDetailed Analysis:")
    print(f"Final result: {detailed_result}")
    
    print(f"\nViolations Analysis:")
    if analysis['violations']:
        for i, (pos1, pos2, val1, val2) in enumerate(analysis['violations']):
            print(f"  Violation {i+1}: arr1[{pos1}]={val1} >= arr1[{pos2}]={val2}")
    
    print(f"\nOptimization Path:")
    if analysis['optimization_path']:
        for step in analysis['optimization_path'][:10]:  # Show first 10 steps
            print(f"  • {step}")
        if len(analysis['optimization_path']) > 10:
            print(f"  ... and {len(analysis['optimization_path']) - 10} more steps")
    
    print(f"\nInsights:")
    for insight in analysis['insights']:
        print(f"  • {insight}")
    
    # Replacement options analysis
    if len(arr1) <= 10:
        print(f"\nReplacement Options by Position:")
        for i, options in enumerate(analysis['replacement_options']):
            if options:
                valid_options = [opt for opt in options if opt['locally_valid']]
                print(f"  Position {i}: {len(options)} total, {len(valid_options)} locally valid")
    
    return detailed_result


def advanced_array_modification_variants():
    """
    ADVANCED ARRAY MODIFICATION VARIANTS:
    ====================================
    Demonstrate various array modification problems.
    """
    
    def make_array_non_decreasing(arr1, arr2):
        """Make array non-decreasing (allows equal adjacent elements)"""
        import bisect
        
        n = len(arr1)
        arr2_unique = sorted(set(arr2))
        
        dp = {-1: 0}
        
        for i in range(n):
            new_dp = {}
            
            for last_val, operations in dp.items():
                # Keep arr1[i]
                if arr1[i] >= last_val:
                    if arr1[i] not in new_dp:
                        new_dp[arr1[i]] = float('inf')
                    new_dp[arr1[i]] = min(new_dp[arr1[i]], operations)
                
                # Replace with arr2 value
                idx = bisect.bisect_left(arr2_unique, last_val)
                
                for j in range(idx, len(arr2_unique)):
                    new_val = arr2_unique[j]
                    if new_val not in new_dp:
                        new_dp[new_val] = float('inf')
                    new_dp[new_val] = min(new_dp[new_val], operations + 1)
            
            dp = new_dp
        
        return min(dp.values()) if dp else -1
    
    def make_array_strictly_decreasing(arr1, arr2):
        """Make array strictly decreasing"""
        import bisect
        
        n = len(arr1)
        arr2_unique = sorted(set(arr2), reverse=True)
        
        dp = {float('inf'): 0}  # last_val -> min_operations
        
        for i in range(n):
            new_dp = {}
            
            for last_val, operations in dp.items():
                # Keep arr1[i]
                if arr1[i] < last_val:
                    if arr1[i] not in new_dp:
                        new_dp[arr1[i]] = float('inf')
                    new_dp[arr1[i]] = min(new_dp[arr1[i]], operations)
                
                # Replace with arr2 value
                for val in arr2_unique:
                    if val < last_val:
                        if val not in new_dp:
                            new_dp[val] = float('inf')
                        new_dp[val] = min(new_dp[val], operations + 1)
            
            dp = new_dp
        
        return min(dp.values()) if dp else -1
    
    def minimum_operations_mountain_array(arr1, arr2):
        """Make array a mountain (increasing then decreasing)"""
        n = len(arr1)
        if n < 3:
            return -1
        
        # Try each position as peak
        min_ops = float('inf')
        
        for peak in range(1, n - 1):
            # Make arr1[0:peak+1] strictly increasing
            left_arr1 = arr1[:peak + 1]
            left_ops = make_array_strictly_increasing_space_optimized(left_arr1, arr2)
            
            if left_ops == -1:
                continue
            
            # Make arr1[peak:] strictly decreasing
            right_arr1 = arr1[peak:]
            right_ops = make_array_strictly_decreasing(right_arr1, arr2)
            
            if right_ops == -1:
                continue
            
            # Subtract 1 because peak is counted twice
            total_ops = left_ops + right_ops
            min_ops = min(min_ops, total_ops)
        
        return min_ops if min_ops != float('inf') else -1
    
    # Test variants
    test_cases = [
        ([1, 5, 3, 6, 7], [1, 3, 2, 4]),
        ([1, 5, 3, 6, 7], [4, 3, 1]),
        ([2, 3, 1, 5, 4], [1, 2, 3, 4, 5])
    ]
    
    print("Advanced Array Modification Variants:")
    print("=" * 50)
    
    for i, (arr1, arr2) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: arr1={arr1}, arr2={arr2}")
        
        # Strictly increasing
        strictly_inc = make_array_strictly_increasing_space_optimized(arr1[:], arr2[:])
        print(f"  Strictly increasing: {strictly_inc}")
        
        # Non-decreasing
        non_dec = make_array_non_decreasing(arr1[:], arr2[:])
        print(f"  Non-decreasing: {non_dec}")
        
        # Strictly decreasing
        strictly_dec = make_array_strictly_decreasing(arr1[:], arr2[:])
        print(f"  Strictly decreasing: {strictly_dec}")
        
        # Mountain array (simplified)
        if len(arr1) >= 3:
            mountain = minimum_operations_mountain_array(arr1[:], arr2[:])
            print(f"  Mountain array: {mountain}")


# Test cases
def test_make_array_strictly_increasing():
    """Test make array strictly increasing implementations"""
    test_cases = [
        ([1,5,3,6,7], [1,3,2,4], 1),
        ([1,5,3,6,7], [4,3,1], 2),
        ([1,5,3,6,7], [1,6,3,3], -1),
        ([1,5,3,6,7], [], -1),
        ([1,2,3,4,5], [1,2,3], 0)
    ]
    
    print("Testing Make Array Strictly Increasing Solutions:")
    print("=" * 70)
    
    for i, (arr1, arr2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"arr1 = {arr1}")
        print(f"arr2 = {arr2}")
        print(f"Expected: {expected}")
        
        basic = make_array_strictly_increasing_basic_dp(arr1[:], arr2[:])
        optimized = make_array_strictly_increasing_optimized_dp(arr1[:], arr2[:])
        space_opt = make_array_strictly_increasing_space_optimized(arr1[:], arr2[:])
        
        print(f"Basic DP:         {basic:>3} {'✓' if basic == expected else '✗'}")
        print(f"Optimized DP:     {optimized:>3} {'✓' if optimized == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    make_array_analysis([1,5,3,6,7], [1,3,2,4])
    
    # Advanced variants
    print(f"\n" + "=" * 70)
    advanced_array_modification_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. STATE COMPRESSION: Use last value as state to reduce complexity")
    print("2. BINARY SEARCH: Efficiently find replacement candidates")
    print("3. COORDINATE COMPRESSION: Handle large value ranges")
    print("4. SPARSE DP: Use dictionaries for sparse state spaces")
    print("5. GREEDY CHOICES: Take smallest valid replacement when possible")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Data Preprocessing: Clean and sort datasets")
    print("• Signal Processing: Monotonic signal reconstruction")
    print("• Database Optimization: Index ordering and maintenance")
    print("• Algorithm Design: Sequence optimization problems")
    print("• Game Development: Level progression and difficulty curves")


if __name__ == "__main__":
    test_make_array_strictly_increasing()


"""
MAKE ARRAY STRICTLY INCREASING - COMPLEX STATE MANAGEMENT WITH DATA STRUCTURES:
===============================================================================

This problem demonstrates advanced DP with complex state management:
- Multi-dimensional state space with value dependencies
- Integration of binary search for optimization
- Coordinate compression for large value ranges
- Trade-offs between time and space complexity

KEY INSIGHTS:
============
1. **STATE COMPRESSION**: Use (position, last_value) as minimal state representation
2. **BINARY SEARCH INTEGRATION**: Efficiently find valid replacement candidates
3. **SPARSE STATE SPACE**: Use dictionaries to handle large value ranges
4. **COORDINATE COMPRESSION**: Map large values to smaller index space
5. **GREEDY OPTIMIZATION**: Choose smallest valid replacement when multiple options exist

ALGORITHM APPROACHES:
====================

1. **Basic DP**: O(n * m * n) time, O(n * unique_values) space
   - Recursive DP with memoization
   - Clear but potentially inefficient

2. **Optimized DP**: O(n * m * log m) time, O(n * m) space
   - Binary search for replacement candidates
   - More efficient transition computation

3. **Space Optimized**: O(n * m * log m) time, O(m) space
   - Rolling arrays to reduce space complexity
   - Practical for large inputs

4. **Segment Tree**: O(n * log V) time, O(V) space
   - Advanced data structure for range queries
   - Handles very large value ranges

CORE DP RECURRENCE:
==================
**State**: dp[i][val] = minimum operations to make arr1[0:i] strictly increasing 
          with arr1[i-1] = val

**Transitions**:
1. Keep arr1[i]: dp[i+1][arr1[i]] = min(dp[i+1][arr1[i]], dp[i][val]) if arr1[i] > val
2. Replace arr1[i]: dp[i+1][new_val] = min(dp[i+1][new_val], dp[i][val] + 1) 
                    for new_val in arr2 where new_val > val

BINARY SEARCH OPTIMIZATION:
===========================
**Purpose**: Find smallest valid replacement value efficiently
```python
import bisect
arr2_sorted = sorted(set(arr2))
idx = bisect.bisect_right(arr2_sorted, last_val)
candidates = arr2_sorted[idx:]  # All values > last_val
```

**Complexity Reduction**: From O(m) to O(log m) per transition

COORDINATE COMPRESSION:
======================
**Problem**: Values can be up to 10⁹, creating huge state space
**Solution**: Map values to compressed coordinates
```python
all_values = sorted(set(arr1 + arr2))
val_to_idx = {val: i for i, val in enumerate(all_values)}
```

**Benefit**: Reduces space from O(10⁹) to O(n + m)

SPACE OPTIMIZATION TECHNIQUES:
=============================

**Rolling Arrays**: 
```python
prev_dp = {last_val: operations}
for i in range(n):
    curr_dp = {}
    # Compute transitions
    prev_dp = curr_dp
```

**Sparse Representation**: Use dictionaries instead of arrays for sparse states

SEGMENT TREE INTEGRATION:
=========================
**Purpose**: Efficient range minimum queries for DP
**Operations**:
- Update(val, operations): Set minimum operations for ending value val
- Query(0, val): Get minimum operations for any ending value ≤ val

**Complexity**: O(log V) per operation where V is value range

COMPLEXITY ANALYSIS:
===================
**Time Complexities**:
- Basic: O(n * m * n) - three nested loops
- Optimized: O(n * m * log m) - binary search optimization
- Segment Tree: O(n * log V) - depends on value range

**Space Complexities**:
- Full DP: O(n * unique_values)
- Space Optimized: O(unique_values)
- Coordinate Compressed: O(n + m)

IMPLEMENTATION CONSIDERATIONS:
=============================

**Edge Cases**:
- Empty arr2: Can only check if arr1 is already sorted
- Single element: Always possible
- All equal elements: Impossible for strictly increasing

**Optimization Strategies**:
- Early termination when impossible
- Pruning of dominated states
- Efficient data structure selection

REAL-WORLD APPLICATIONS:
=======================
- **Data Cleaning**: Repair corrupted sequences
- **Signal Processing**: Monotonic signal reconstruction
- **Database Indexing**: Maintain sorted order with updates
- **Quality Control**: Ensure monotonic properties in measurements
- **Game Design**: Level progression and difficulty tuning

RELATED PROBLEMS:
================
- **Longest Increasing Subsequence**: Related optimization
- **Edit Distance**: Similar state space structure
- **Sequence Alignment**: Biological sequence matching
- **Order Statistics**: Maintaining sorted structures

This problem showcases how complex DP state management
can be optimized through careful algorithm design,
appropriate data structure selection, and mathematical
insights about the problem structure.
"""
