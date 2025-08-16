"""
LeetCode 1130: Minimum Cost Tree From Leaf Values
Difficulty: Medium
Category: Interval DP - Tree Construction

PROBLEM DESCRIPTION:
===================
Given an array arr of positive integers, consider all binary trees such that:
- Each node has either 0 or 2 children;
- The values of arr correspond to the values of each leaf of some binary tree;
- The value of each non-leaf node is equal to the product of the largest leaf value in its left and right subtree respectively.

Among all possible binary trees considered, return the smallest possible sum of values of all non-leaf nodes. 
It is guaranteed that the answer fits in a 32-bit signed integer.

Example 1:
Input: arr = [6,2,4]
Output: 32
Explanation: There are two possible trees shown.
The first has a non-leaf node sum 6*4 + 6*2 = 32.
The second has a non-leaf node sum 2*4 + 6*4 = 32.

Example 2:
Input: arr = [4,11]
Output: 44
Explanation: The only possible tree has a non-leaf node with value 4*11 = 44.

Constraints:
- 2 <= arr.length <= 40
- 1 <= arr[i] <= 15
- It is guaranteed that the answer fits in a 32-bit signed integer.
"""

def mct_from_leaf_values_brute_force(arr):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible binary tree structures.
    
    Time Complexity: O(C_n) - Catalan number, exponential
    Space Complexity: O(n) - recursion depth
    """
    def build_tree(start, end):
        # Base case: single leaf
        if start == end:
            return 0, arr[start]  # (cost, max_value)
        
        min_cost = float('inf')
        max_val = max(arr[start:end+1])
        
        # Try all possible split points
        for k in range(start, end):
            left_cost, left_max = build_tree(start, k)
            right_cost, right_max = build_tree(k + 1, end)
            
            # Cost of current internal node
            internal_cost = left_max * right_max
            total_cost = left_cost + right_cost + internal_cost
            
            min_cost = min(min_cost, total_cost)
        
        return min_cost, max_val
    
    cost, _ = build_tree(0, len(arr) - 1)
    return cost


def mct_from_leaf_values_memoization(arr):
    """
    MEMOIZATION APPROACH:
    ====================
    Cache results for subarray ranges.
    
    Time Complexity: O(n^3) - with memoization
    Space Complexity: O(n^2) - memo table
    """
    memo = {}
    
    def dp(start, end):
        if start == end:
            return 0
        
        if (start, end) in memo:
            return memo[(start, end)]
        
        min_cost = float('inf')
        
        for k in range(start, end):
            left_max = max(arr[start:k+1])
            right_max = max(arr[k+1:end+1])
            
            left_cost = dp(start, k)
            right_cost = dp(k + 1, end)
            
            total_cost = left_cost + right_cost + left_max * right_max
            min_cost = min(min_cost, total_cost)
        
        memo[(start, end)] = min_cost
        return min_cost
    
    return dp(0, len(arr) - 1)


def mct_from_leaf_values_interval_dp(arr):
    """
    INTERVAL DP APPROACH:
    ====================
    Bottom-up DP with precomputed max values.
    
    Time Complexity: O(n^3) - three nested loops
    Space Complexity: O(n^2) - DP table and max table
    """
    n = len(arr)
    
    # Precompute max values for all subarrays
    max_val = [[0] * n for _ in range(n)]
    for i in range(n):
        max_val[i][i] = arr[i]
        for j in range(i + 1, n):
            max_val[i][j] = max(max_val[i][j-1], arr[j])
    
    # dp[i][j] = minimum cost for subarray arr[i:j+1]
    dp = [[0] * n for _ in range(n)]
    
    # Process intervals by length
    for length in range(2, n + 1):  # At least 2 leaves for internal node
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            # Try all split points
            for k in range(i, j):
                left_max = max_val[i][k]
                right_max = max_val[k+1][j]
                
                cost = dp[i][k] + dp[k+1][j] + left_max * right_max
                dp[i][j] = min(dp[i][j], cost)
    
    return dp[0][n-1]


def mct_from_leaf_values_monotonic_stack(arr):
    """
    MONOTONIC STACK APPROACH:
    ========================
    Greedy approach using stack to find optimal merging order.
    
    Time Complexity: O(n^2) - each element pushed/popped once, inner work O(n)
    Space Complexity: O(n) - stack space
    """
    stack = [float('inf')]  # Sentinel value
    result = 0
    
    for val in arr + [float('inf')]:  # Add sentinel at end
        while stack[-1] <= val:
            # Remove elements that are <= current value
            mid = stack.pop()
            # Cost of merging: mid with min(left, right)
            result += mid * min(stack[-1], val)
        
        stack.append(val)
    
    return result


def mct_from_leaf_values_with_tree(arr):
    """
    TRACK ACTUAL TREE STRUCTURE:
    ============================
    Return minimum cost and the actual binary tree structure.
    
    Time Complexity: O(n^3) - DP computation + reconstruction
    Space Complexity: O(n^2) - DP table + tree tracking
    """
    n = len(arr)
    
    # Precompute max values
    max_val = [[0] * n for _ in range(n)]
    for i in range(n):
        max_val[i][i] = arr[i]
        for j in range(i + 1, n):
            max_val[i][j] = max(max_val[i][j-1], arr[j])
    
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]  # Track optimal split points
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            for k in range(i, j):
                left_max = max_val[i][k]
                right_max = max_val[k+1][j]
                cost = dp[i][k] + dp[k+1][j] + left_max * right_max
                
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    
    # Reconstruct tree structure
    def build_tree_structure(start, end):
        if start == end:
            return {'type': 'leaf', 'value': arr[start], 'index': start}
        
        k = split[start][end]
        left_subtree = build_tree_structure(start, k)
        right_subtree = build_tree_structure(k + 1, end)
        
        left_max = max_val[start][k]
        right_max = max_val[k+1][end]
        
        return {
            'type': 'internal',
            'value': left_max * right_max,
            'left': left_subtree,
            'right': right_subtree
        }
    
    tree = build_tree_structure(0, n - 1)
    return dp[0][n-1], tree


def mct_from_leaf_values_analysis(arr):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and tree construction.
    """
    print(f"Minimum Cost Tree Analysis:")
    print(f"Leaf values: {arr}")
    
    n = len(arr)
    
    # Show max value precomputation
    max_val = [[0] * n for _ in range(n)]
    for i in range(n):
        max_val[i][i] = arr[i]
        for j in range(i + 1, n):
            max_val[i][j] = max(max_val[i][j-1], arr[j])
    
    print(f"\nMax value table:")
    print("   ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            if j >= i:
                print(f"{max_val[i][j]:4}", end="")
            else:
                print(f"{'':4}", end="")
        print()
    
    # Build DP table with detailed logging
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    
    print(f"\nDP Table Construction:")
    
    for length in range(2, n + 1):
        print(f"\nLength {length} intervals:")
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            print(f"  Interval [{i}, {j}] - leaves: {arr[i:j+1]}")
            
            for k in range(i, j):
                left_max = max_val[i][k]
                right_max = max_val[k+1][j]
                cost = dp[i][k] + dp[k+1][j] + left_max * right_max
                
                print(f"    Split at {k}: left[{i},{k}] right[{k+1},{j}]")
                print(f"      Left max: {left_max}, Right max: {right_max}")
                print(f"      Internal node cost: {left_max} * {right_max} = {left_max * right_max}")
                print(f"      Total: {dp[i][k]} + {dp[k+1][j]} + {left_max * right_max} = {cost}")
                
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
                    print(f"      *** New minimum for [{i},{j}]")
            
            print(f"    Best for [{i},{j}]: {dp[i][j]} (split at {split[i][j]})")
    
    print(f"\nFinal DP Table:")
    print("   ", end="")
    for j in range(n):
        print(f"{j:6}", end="")
    print()
    
    for i in range(n):
        print(f"{i:2}: ", end="")
        for j in range(n):
            if j >= i:
                print(f"{dp[i][j]:6}", end="")
            else:
                print(f"{'':6}", end="")
        print()
    
    print(f"\nMinimum cost: {dp[0][n-1]}")
    
    # Show tree structure
    min_cost, tree = mct_from_leaf_values_with_tree(arr)
    print(f"\nOptimal tree structure:")
    
    def print_tree(node, depth=0):
        indent = "  " * depth
        if node['type'] == 'leaf':
            print(f"{indent}Leaf: {node['value']} (index {node['index']})")
        else:
            print(f"{indent}Internal: {node['value']}")
            print(f"{indent}├─ Left:")
            print_tree(node['left'], depth + 1)
            print(f"{indent}└─ Right:")
            print_tree(node['right'], depth + 1)
    
    print_tree(tree)
    
    # Compare with monotonic stack approach
    stack_result = mct_from_leaf_values_monotonic_stack(arr)
    print(f"\nMonotonic stack result: {stack_result}")
    print(f"Results match: {min_cost == stack_result}")
    
    return dp[0][n-1]


def mct_from_leaf_values_variants():
    """
    MINIMUM COST TREE VARIANTS:
    ===========================
    Different scenarios and modifications.
    """
    
    def max_cost_tree(arr):
        """Find maximum cost tree instead of minimum"""
        n = len(arr)
        max_val = [[0] * n for _ in range(n)]
        
        for i in range(n):
            max_val[i][i] = arr[i]
            for j in range(i + 1, n):
                max_val[i][j] = max(max_val[i][j-1], arr[j])
        
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                for k in range(i, j):
                    left_max = max_val[i][k]
                    right_max = max_val[k+1][j]
                    cost = dp[i][k] + dp[k+1][j] + left_max * right_max
                    dp[i][j] = max(dp[i][j], cost)
        
        return dp[0][n-1]
    
    def mct_with_min_operation(arr):
        """Use min instead of max for internal node values"""
        n = len(arr)
        min_val = [[float('inf')] * n for _ in range(n)]
        
        for i in range(n):
            min_val[i][i] = arr[i]
            for j in range(i + 1, n):
                min_val[i][j] = min(min_val[i][j-1], arr[j])
        
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    left_min = min_val[i][k]
                    right_min = min_val[k+1][j]
                    cost = dp[i][k] + dp[k+1][j] + left_min * right_min
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n-1]
    
    def count_possible_trees(n):
        """Count number of possible binary trees (Catalan number)"""
        if n <= 1:
            return 1
        
        def catalan(k):
            if k <= 1:
                return 1
            
            result = 1
            for i in range(k):
                result = result * (k + 1 + i) // (i + 1)
            return result // (k + 1)
        
        return catalan(n - 1)
    
    def mct_with_weights(arr, weights):
        """Each leaf has a weight affecting the cost calculation"""
        n = len(arr)
        
        # Weighted max calculation
        weighted_max = [[0] * n for _ in range(n)]
        for i in range(n):
            weighted_max[i][i] = arr[i] * weights[i]
            for j in range(i + 1, n):
                if arr[j] * weights[j] > weighted_max[i][j-1]:
                    weighted_max[i][j] = arr[j] * weights[j]
                else:
                    weighted_max[i][j] = weighted_max[i][j-1]
        
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                
                for k in range(i, j):
                    left_max = weighted_max[i][k]
                    right_max = weighted_max[k+1][j]
                    cost = dp[i][k] + dp[k+1][j] + left_max * right_max
                    dp[i][j] = min(dp[i][j], cost)
        
        return dp[0][n-1]
    
    # Test variants
    test_cases = [
        [6, 2, 4],
        [4, 11],
        [1, 2, 3, 4, 5],
        [5, 3, 1, 4, 2]
    ]
    
    print("Minimum Cost Tree Variants:")
    print("=" * 50)
    
    for arr in test_cases:
        print(f"\nLeaf values: {arr}")
        
        min_cost = mct_from_leaf_values_interval_dp(arr)
        max_cost = max_cost_tree(arr)
        min_operation = mct_with_min_operation(arr)
        tree_count = count_possible_trees(len(arr))
        
        print(f"Min cost (max operation): {min_cost}")
        print(f"Max cost (max operation): {max_cost}")
        print(f"Min cost (min operation): {min_operation}")
        print(f"Possible tree structures: {tree_count}")
        
        # With weights
        weights = [1.5] * len(arr)
        weighted_cost = mct_with_weights(arr, weights)
        print(f"With 1.5x weights: {weighted_cost}")


# Test cases
def test_mct_from_leaf_values():
    """Test all implementations with various inputs"""
    test_cases = [
        ([6, 2, 4], 32),
        ([4, 11], 44),
        ([1, 2, 3], 4),
        ([2, 3, 1, 4], 12),
        ([1, 2, 3, 4, 5], 44),
        ([5, 3, 1, 4, 2], 47)
    ]
    
    print("Testing Minimum Cost Tree From Leaf Values Solutions:")
    print("=" * 70)
    
    for i, (arr, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Array: {arr}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large inputs
        if len(arr) <= 5:
            try:
                brute_force = mct_from_leaf_values_brute_force(arr)
                print(f"Brute Force:      {brute_force:>4} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memoization = mct_from_leaf_values_memoization(arr)
        interval_dp = mct_from_leaf_values_interval_dp(arr)
        monotonic_stack = mct_from_leaf_values_monotonic_stack(arr)
        
        print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        print(f"Interval DP:      {interval_dp:>4} {'✓' if interval_dp == expected else '✗'}")
        print(f"Monotonic Stack:  {monotonic_stack:>4} {'✓' if monotonic_stack == expected else '✗'}")
        
        # Show tree structure for small cases
        if len(arr) <= 5:
            min_cost, tree = mct_from_leaf_values_with_tree(arr)
            print(f"Tree structure cost: {min_cost}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    mct_from_leaf_values_analysis([6, 2, 4])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    mct_from_leaf_values_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. INTERVAL DP: Optimal tree structure via optimal split points")
    print("2. INTERNAL NODES: Value = max(left_subtree) * max(right_subtree)")
    print("3. MONOTONIC STACK: O(n^2) greedy approach for optimization")
    print("4. CATALAN NUMBERS: Number of possible tree structures")
    print("5. PRECOMPUTATION: Max values for all subarrays speeds up DP")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Compiler Design: Expression tree optimization")
    print("• Database Systems: Query execution plan optimization")
    print("• Algorithm Design: Tree construction optimization")
    print("• Computational Biology: Phylogenetic tree construction")
    print("• Data Structures: Optimal search tree construction")


if __name__ == "__main__":
    test_mct_from_leaf_values()


"""
MINIMUM COST TREE FROM LEAF VALUES - OPTIMAL TREE CONSTRUCTION:
===============================================================

This problem combines interval DP with binary tree construction optimization:
- Build binary tree where leaves are given array values
- Internal node value = max(left subtree) × max(right subtree)
- Minimize sum of all internal node values
- Demonstrates interval DP for hierarchical structure optimization

KEY INSIGHTS:
============
1. **TREE STRUCTURE**: Each internal node has exactly 2 children
2. **NODE VALUE RULE**: Internal node = max(left) × max(right)
3. **OPTIMIZATION TARGET**: Minimize sum of internal node values
4. **INTERVAL SPLITTING**: Each split point determines tree structure
5. **GREEDY ALTERNATIVE**: Monotonic stack provides O(n²) solution

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(C_n) time, O(n) space
   - Try all possible binary tree structures
   - Catalan number complexity

2. **Memoization**: O(n³) time, O(n²) space
   - Top-down DP with caching
   - Natural recursive structure

3. **Interval DP**: O(n³) time, O(n²) space
   - Bottom-up construction
   - Standard approach with precomputed max values

4. **Monotonic Stack**: O(n²) time, O(n) space
   - Greedy approach using stack
   - Optimal for this specific problem

CORE INTERVAL DP ALGORITHM:
==========================
```python
# Precompute max values for all subarrays
max_val[i][j] = max(arr[i:j+1])

# dp[i][j] = minimum cost for subarray arr[i:j+1]
for length in range(2, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        
        for k in range(i, j):  # Split point
            left_max = max_val[i][k]
            right_max = max_val[k+1][j]
            
            cost = dp[i][k] + dp[k+1][j] + left_max * right_max
            dp[i][j] = min(dp[i][j], cost)
```

MONOTONIC STACK OPTIMIZATION:
============================
Key insight: Always merge smaller elements first
```python
stack = [infinity]  # Sentinel
result = 0

for val in arr + [infinity]:
    while stack[-1] <= val:
        mid = stack.pop()
        result += mid * min(stack[-1], val)
    stack.append(val)

return result
```

**Why this works**: 
- Greedy choice: always merge the smallest available element
- Stack maintains decreasing order
- Each merge cost = mid × min(left_neighbor, right_neighbor)

TREE CONSTRUCTION PROPERTIES:
============================
**Binary Tree Constraints**:
- Each internal node has exactly 2 children
- Leaves correspond to array elements in order
- Tree structure determines the merging order

**Node Value Calculation**:
- Leaf node: original array value
- Internal node: max(left_subtree) × max(right_subtree)
- Total cost: sum of all internal node values

RECURRENCE RELATION:
===================
```
dp[i][j] = min(dp[i][k] + dp[k+1][j] + max_val[i][k] * max_val[k+1][j])
           for all k in [i, j-1]

Base case: dp[i][i] = 0 (single leaf, no internal nodes)
```

**Intuition**: To build optimal tree for arr[i:j+1]:
- Choose split point k to divide into arr[i:k+1] and arr[k+1:j+1]
- Recursively build optimal subtrees
- Add cost of root: max(left_subtree) × max(right_subtree)

COMPLEXITY ANALYSIS:
===================
| Approach         | Time  | Space | Notes                    |
|------------------|-------|-------|--------------------------|
| Brute Force      | O(C_n)| O(n)  | Catalan number growth    |
| Memoization      | O(n³) | O(n²) | Top-down with caching    |
| Interval DP      | O(n³) | O(n²) | Bottom-up standard       |
| Monotonic Stack  | O(n²) | O(n)  | Greedy optimization      |

PRECOMPUTATION OPTIMIZATION:
===========================
Computing max values for all subarrays upfront:
```python
max_val = [[0] * n for _ in range(n)]
for i in range(n):
    max_val[i][i] = arr[i]
    for j in range(i + 1, n):
        max_val[i][j] = max(max_val[i][j-1], arr[j])
```

This avoids O(n) computation for each DP state transition.

SOLUTION RECONSTRUCTION:
=======================
To build the actual tree structure:
```python
def build_tree(start, end, split):
    if start == end:
        return {'type': 'leaf', 'value': arr[start]}
    
    k = split[start][end]
    return {
        'type': 'internal',
        'value': max_val[start][k] * max_val[k+1][end],
        'left': build_tree(start, k, split),
        'right': build_tree(k+1, end, split)
    }
```

MATHEMATICAL PROPERTIES:
========================
- **Optimal Substructure**: Optimal tree contains optimal subtrees
- **Overlapping Subproblems**: Same subarrays processed multiple times
- **Monotonicity**: Larger arrays have more tree structure options
- **Greedy Choice**: For this problem, smallest-first merging is optimal

APPLICATIONS:
============
- **Compiler Optimization**: Expression tree construction
- **Database Query Planning**: Optimal join order determination
- **Huffman Coding**: Optimal prefix code tree construction
- **Computational Biology**: Phylogenetic tree construction
- **Algorithm Design**: Hierarchical clustering optimization

RELATED PROBLEMS:
================
- **Burst Balloons (312)**: Similar interval DP pattern
- **Matrix Chain Multiplication**: Classic interval optimization
- **Optimal Binary Search Trees**: Weighted tree construction
- **Merge Stones (1000)**: Generalized merging with constraints

GREEDY CORRECTNESS:
==================
For this specific problem, the monotonic stack approach is optimal because:
1. **Local Optimality**: Always merge the smallest available element
2. **No Regret**: Early merging of small elements never hurts
3. **Structural Independence**: Merge order doesn't affect final tree structure optimality

EDGE CASES:
==========
- **Two elements**: Only one possible tree structure
- **Sorted array**: Stack approach is most efficient
- **Reverse sorted**: Worst case for some approaches
- **All equal elements**: Multiple optimal solutions

OPTIMIZATION TECHNIQUES:
=======================
- **Early Termination**: Prune suboptimal branches in brute force
- **Space Optimization**: Reduce DP table memory usage
- **Parallel Processing**: Independent subproblems in interval DP
- **Approximation**: Heuristics for very large inputs

This problem beautifully demonstrates how interval DP can solve complex
tree construction problems, while also showing how problem-specific
insights can lead to more efficient greedy algorithms.
"""
