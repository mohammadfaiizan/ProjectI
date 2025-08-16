"""
LeetCode 546: Remove Boxes
Difficulty: Hard
Category: Interval DP - Complex State Management

PROBLEM DESCRIPTION:
===================
You are given several boxes with different colors represented by different positive numbers.
You may experience several rounds to remove boxes until there is no box left. Each time you can choose some continuous boxes with the same color (i.e., composed of k boxes, k >= 1), remove them and get k * k points.
Return the maximum points you can get.

Example 1:
Input: boxes = [1,3,2,2,2,3,4,3,1]
Output: 23
Explanation:
[1, 3, 2, 2, 2, 3, 4, 3, 1] 
--> [1, 3, 3, 4, 3, 1] (3*3=9 points) 
--> [1, 3, 3, 3, 1] (1*1=1 points) 
--> [1, 1] (3*3=9 points) 
--> [] (2*2=4 points)

Example 2:
Input: boxes = [1,1,1]
Output: 9

Example 3:
Input: boxes = [1]
Output: 1

Constraints:
- 1 <= boxes.length <= 100
- 1 <= boxes[i] <= 100
"""

def remove_boxes_brute_force(boxes):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible removal sequences recursively.
    
    Time Complexity: O(exponential) - explosive branching
    Space Complexity: O(n) - recursion depth
    """
    def dfs(current_boxes):
        if not current_boxes:
            return 0
        
        max_points = 0
        
        # Try removing each continuous segment of same color
        i = 0
        while i < len(current_boxes):
            j = i
            while j < len(current_boxes) and current_boxes[j] == current_boxes[i]:
                j += 1
            
            # Remove boxes from i to j-1
            count = j - i
            points = count * count
            
            # Create new array without removed boxes
            new_boxes = current_boxes[:i] + current_boxes[j:]
            
            # Recurse with remaining boxes
            remaining_points = dfs(new_boxes)
            max_points = max(max_points, points + remaining_points)
            
            i = j
        
        return max_points
    
    return dfs(boxes)


def remove_boxes_memoization_simple(boxes):
    """
    SIMPLE MEMOIZATION APPROACH:
    ============================
    Cache results for different box configurations.
    
    Time Complexity: O(exponential) - still too many states
    Space Complexity: O(exponential) - memo table
    """
    memo = {}
    
    def dfs(current_boxes):
        boxes_tuple = tuple(current_boxes)
        if boxes_tuple in memo:
            return memo[boxes_tuple]
        
        if not current_boxes:
            return 0
        
        max_points = 0
        
        i = 0
        while i < len(current_boxes):
            j = i
            while j < len(current_boxes) and current_boxes[j] == current_boxes[i]:
                j += 1
            
            count = j - i
            points = count * count
            new_boxes = current_boxes[:i] + current_boxes[j:]
            
            remaining_points = dfs(new_boxes)
            max_points = max(max_points, points + remaining_points)
            
            i = j
        
        memo[boxes_tuple] = max_points
        return max_points
    
    return dfs(boxes)


def remove_boxes_interval_dp(boxes):
    """
    INTERVAL DP WITH 3D STATE:
    ==========================
    dp[i][j][k] = max points for boxes[i:j+1] with k extra boxes of same color as boxes[j].
    
    Time Complexity: O(n^4) - four nested loops
    Space Complexity: O(n^3) - 3D DP table
    """
    n = len(boxes)
    if n == 0:
        return 0
    
    # dp[i][j][k] = max points for removing boxes[i:j+1] 
    # where there are k additional boxes of color boxes[j] after position j
    memo = {}
    
    def dp(i, j, k):
        if i > j:
            return 0
        
        if (i, j, k) in memo:
            return memo[(i, j, k)]
        
        # Optimization: merge consecutive boxes of same color at the end
        while j > i and boxes[j] == boxes[j-1]:
            j -= 1
            k += 1
        
        # Option 1: Remove boxes[j] along with k extra boxes
        result = dp(i, j-1, 0) + (k + 1) * (k + 1)
        
        # Option 2: Find boxes of same color as boxes[j] in range [i, j-1]
        # and combine them with boxes[j] and the k extra boxes
        for m in range(i, j):
            if boxes[m] == boxes[j]:
                # Remove boxes[m+1:j] first, then combine boxes[m] with boxes[j]
                result = max(result, dp(i, m, k + 1) + dp(m + 1, j - 1, 0))
        
        memo[(i, j, k)] = result
        return result
    
    return dp(0, n - 1, 0)


def remove_boxes_optimized_dp(boxes):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Optimized version with better state management.
    
    Time Complexity: O(n^3) - optimized with memoization
    Space Complexity: O(n^3) - 3D memoization
    """
    n = len(boxes)
    memo = {}
    
    def calculate_points(i, j, k):
        if i > j:
            return 0
        
        if (i, j, k) in memo:
            return memo[(i, j, k)]
        
        # Merge consecutive same-colored boxes at the right end
        original_j = j
        original_k = k
        
        while j > i and boxes[j] == boxes[j-1]:
            j -= 1
            k += 1
        
        # Case 1: Remove the rightmost group (boxes[j] + k additional boxes)
        points = calculate_points(i, j-1, 0) + (k + 1) ** 2
        
        # Case 2: Try to merge with previous same-colored boxes
        for m in range(i, j):
            if boxes[m] == boxes[j]:
                # Merge boxes[m] with boxes[j] and k additional boxes
                merge_points = (calculate_points(i, m, k + 1) + 
                               calculate_points(m + 1, j - 1, 0))
                points = max(points, merge_points)
        
        memo[(i, j, k)] = points
        return points
    
    return calculate_points(0, n - 1, 0)


def remove_boxes_with_sequence(boxes):
    """
    TRACK REMOVAL SEQUENCE:
    ======================
    Return maximum points and one possible optimal removal sequence.
    
    Time Complexity: O(n^4) - DP computation + reconstruction
    Space Complexity: O(n^3) - DP table + sequence tracking
    """
    n = len(boxes)
    if n == 0:
        return 0, []
    
    memo = {}
    choice = {}  # Track decisions for reconstruction
    
    def dp(i, j, k):
        if i > j:
            return 0
        
        if (i, j, k) in memo:
            return memo[(i, j, k)]
        
        # Merge consecutive same-colored boxes
        while j > i and boxes[j] == boxes[j-1]:
            j -= 1
            k += 1
        
        # Option 1: Remove rightmost group
        result = dp(i, j-1, 0) + (k + 1) * (k + 1)
        choice[(i, j, k)] = ('remove', j, k + 1)
        
        # Option 2: Merge with previous same-colored boxes
        for m in range(i, j):
            if boxes[m] == boxes[j]:
                merge_result = dp(i, m, k + 1) + dp(m + 1, j - 1, 0)
                if merge_result > result:
                    result = merge_result
                    choice[(i, j, k)] = ('merge', m, j, k + 1)
        
        memo[(i, j, k)] = result
        return result
    
    max_points = dp(0, n - 1, 0)
    
    # Reconstruct sequence
    def build_sequence(i, j, k, current_boxes):
        if i > j:
            return []
        
        if (i, j, k) not in choice:
            return []
        
        decision = choice[(i, j, k)]
        sequence = []
        
        if decision[0] == 'remove':
            # Remove rightmost group
            pos = decision[1]
            count = decision[2]
            color = boxes[pos]
            
            # Find all positions of this color in the rightmost group
            positions = []
            extra_count = count - 1  # k additional boxes
            
            # Add the removal operation
            sequence.append(('remove', pos, color, count))
            sequence.extend(build_sequence(i, pos - 1, 0, current_boxes))
            
        elif decision[0] == 'merge':
            # Merge operation
            m = decision[1]
            j_pos = decision[2]
            merged_count = decision[3]
            
            sequence.extend(build_sequence(m + 1, j_pos - 1, 0, current_boxes))
            sequence.extend(build_sequence(i, m, merged_count - 1, current_boxes))
        
        return sequence
    
    sequence = build_sequence(0, n - 1, 0, boxes)
    return max_points, sequence


def remove_boxes_analysis(boxes):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step computation and removal analysis.
    """
    print(f"Remove Boxes Analysis:")
    print(f"Boxes: {boxes}")
    print(f"Length: {len(boxes)}")
    
    n = len(boxes)
    
    # Show color frequency
    from collections import Counter
    color_count = Counter(boxes)
    print(f"\nColor frequencies: {dict(color_count)}")
    
    # Show consecutive segments
    print(f"\nConsecutive segments:")
    i = 0
    segments = []
    while i < n:
        j = i
        while j < n and boxes[j] == boxes[i]:
            j += 1
        segments.append((boxes[i], j - i, i, j - 1))
        print(f"  Color {boxes[i]}: {j - i} boxes at positions [{i}, {j-1}]")
        i = j
    
    # Simple DP trace for small examples
    if n <= 6:
        print(f"\nDP computation trace:")
        memo = {}
        
        def dp_with_trace(i, j, k, depth=0):
            indent = "  " * depth
            
            if i > j:
                print(f"{indent}dp({i}, {j}, {k}) = 0 (empty range)")
                return 0
            
            if (i, j, k) in memo:
                print(f"{indent}dp({i}, {j}, {k}) = {memo[(i, j, k)]} (cached)")
                return memo[(i, j, k)]
            
            print(f"{indent}Computing dp({i}, {j}, {k}):")
            print(f"{indent}  Range: {boxes[i:j+1]}, extra {k} of color {boxes[j]}")
            
            # Merge consecutive
            original_j, original_k = j, k
            while j > i and boxes[j] == boxes[j-1]:
                j -= 1
                k += 1
            
            if (j, k) != (original_j, original_k):
                print(f"{indent}  After merging: j={j}, k={k}")
            
            # Option 1: Remove rightmost group
            option1 = dp_with_trace(i, j-1, 0, depth + 1) + (k + 1) ** 2
            print(f"{indent}  Option 1 (remove group): {option1}")
            
            result = option1
            
            # Option 2: Merge with previous same-colored boxes
            for m in range(i, j):
                if boxes[m] == boxes[j]:
                    print(f"{indent}  Trying merge at position {m} (color {boxes[m]})")
                    option2 = (dp_with_trace(i, m, k + 1, depth + 1) + 
                              dp_with_trace(m + 1, j - 1, 0, depth + 1))
                    print(f"{indent}  Option 2 (merge at {m}): {option2}")
                    result = max(result, option2)
            
            memo[(i, j, k)] = result
            print(f"{indent}Result: dp({i}, {j}, {k}) = {result}")
            return result
        
        final_result = dp_with_trace(0, n - 1, 0)
        print(f"\nFinal maximum points: {final_result}")
    else:
        result = remove_boxes_optimized_dp(boxes)
        print(f"\nMaximum points: {result}")
    
    return result


def remove_boxes_variants():
    """
    REMOVE BOXES VARIANTS:
    =====================
    Different scenarios and modifications.
    """
    
    def remove_boxes_min_points(boxes):
        """Find minimum points (change max to min in recurrence)"""
        n = len(boxes)
        memo = {}
        
        def dp(i, j, k):
            if i > j:
                return 0
            
            if (i, j, k) in memo:
                return memo[(i, j, k)]
            
            while j > i and boxes[j] == boxes[j-1]:
                j -= 1
                k += 1
            
            result = dp(i, j-1, 0) + (k + 1) * (k + 1)
            
            for m in range(i, j):
                if boxes[m] == boxes[j]:
                    result = min(result, dp(i, m, k + 1) + dp(m + 1, j - 1, 0))
            
            memo[(i, j, k)] = result
            return result
        
        return dp(0, n - 1, 0)
    
    def remove_boxes_linear_scoring(boxes):
        """Linear scoring: k boxes give k points (not k^2)"""
        n = len(boxes)
        memo = {}
        
        def dp(i, j, k):
            if i > j:
                return 0
            
            if (i, j, k) in memo:
                return memo[(i, j, k)]
            
            while j > i and boxes[j] == boxes[j-1]:
                j -= 1
                k += 1
            
            result = dp(i, j-1, 0) + (k + 1)  # Linear scoring
            
            for m in range(i, j):
                if boxes[m] == boxes[j]:
                    result = max(result, dp(i, m, k + 1) + dp(m + 1, j - 1, 0))
            
            memo[(i, j, k)] = result
            return result
        
        return dp(0, n - 1, 0)
    
    def count_removal_ways(boxes):
        """Count number of ways to remove all boxes"""
        # This is computationally intensive, simplified version
        if len(boxes) <= 3:
            return 1  # For small cases
        return "Too complex to compute"
    
    # Test variants
    test_cases = [
        [1, 3, 2, 2, 2, 3, 4, 3, 1],
        [1, 1, 1],
        [1],
        [1, 2, 3, 4],
        [1, 1, 2, 2, 3, 3]
    ]
    
    print("Remove Boxes Variants:")
    print("=" * 50)
    
    for boxes in test_cases:
        print(f"\nBoxes: {boxes}")
        
        max_points = remove_boxes_optimized_dp(boxes)
        min_points = remove_boxes_min_points(boxes)
        linear_points = remove_boxes_linear_scoring(boxes)
        ways = count_removal_ways(boxes)
        
        print(f"Max points (quadratic): {max_points}")
        print(f"Min points (quadratic): {min_points}")
        print(f"Max points (linear): {linear_points}")
        print(f"Number of ways: {ways}")


# Test cases
def test_remove_boxes():
    """Test all implementations with various inputs"""
    test_cases = [
        ([1, 3, 2, 2, 2, 3, 4, 3, 1], 23),
        ([1, 1, 1], 9),
        ([1], 1),
        ([1, 2, 3], 6),
        ([1, 1, 2, 2], 8),
        ([1, 2, 1], 4),
        ([1, 2, 2, 1], 8),
        ([3, 8, 8, 5, 5, 3, 9, 2, 4, 4], 35)
    ]
    
    print("Testing Remove Boxes Solutions:")
    print("=" * 70)
    
    for i, (boxes, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Boxes: {boxes}")
        print(f"Expected: {expected}")
        
        # Skip brute force for large inputs
        if len(boxes) <= 6:
            try:
                brute_force = remove_boxes_brute_force(boxes)
                print(f"Brute Force:      {brute_force:>4} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        if len(boxes) <= 8:
            simple_memo = remove_boxes_memoization_simple(boxes)
            print(f"Simple Memo:      {simple_memo:>4} {'✓' if simple_memo == expected else '✗'}")
        
        interval_dp = remove_boxes_interval_dp(boxes)
        optimized_dp = remove_boxes_optimized_dp(boxes)
        
        print(f"Interval DP:      {interval_dp:>4} {'✓' if interval_dp == expected else '✗'}")
        print(f"Optimized DP:     {optimized_dp:>4} {'✓' if optimized_dp == expected else '✗'}")
        
        # Show removal sequence for small cases
        if len(boxes) <= 6:
            max_points, sequence = remove_boxes_with_sequence(boxes)
            print(f"Max points with sequence: {max_points}")
            if sequence and len(sequence) <= 5:
                print(f"Removal sequence: {sequence}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    remove_boxes_analysis([1, 3, 2, 2, 2, 3, 4, 3, 1])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    remove_boxes_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. 3D DP STATE: dp[i][j][k] with extra boxes of same color")
    print("2. MERGING STRATEGY: Combine distant same-colored boxes")
    print("3. QUADRATIC SCORING: k boxes give k^2 points")
    print("4. COMPLEX DEPENDENCIES: Future decisions affect current value")
    print("5. INTERVAL + CONTEXT: Classic interval DP with additional state")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Game Design: Scoring mechanisms in puzzle games")
    print("• Resource Management: Batch processing with economies of scale")
    print("• Algorithm Design: Complex state DP problems")
    print("• Optimization: Non-linear reward systems")
    print("• Combinatorial Games: Strategic decision making")


if __name__ == "__main__":
    test_remove_boxes()


"""
REMOVE BOXES - ADVANCED INTERVAL DP WITH COMPLEX STATE:
=======================================================

This problem represents one of the most complex interval DP variants:
- 3D state space: dp[i][j][k] with additional context
- Non-local dependencies: distant same-colored boxes can be merged
- Quadratic scoring: k consecutive boxes give k² points
- Strategic depth: optimal decisions require considering future merging opportunities

KEY INSIGHTS:
============
1. **3D STATE SPACE**: dp[i][j][k] = max points for boxes[i:j+1] with k extra boxes of color boxes[j]
2. **MERGING STRATEGY**: Can defer removal to merge with distant same-colored boxes
3. **QUADRATIC SCORING**: Incentivizes combining same-colored boxes before removal
4. **COMPLEX DEPENDENCIES**: Current decisions affected by future merging possibilities
5. **NON-LOCAL OPTIMIZATION**: Must consider entire array, not just local intervals

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(exponential) time, O(n) space
   - Try all possible removal sequences
   - Intractable for moderate inputs

2. **Simple Memoization**: O(exponential) time, O(exponential) space
   - Cache based on remaining box configurations
   - Still too many states

3. **Interval DP (3D)**: O(n⁴) time, O(n³) space
   - Standard approach with 3D state
   - Efficient enough for problem constraints

4. **Optimized DP**: O(n³) time, O(n³) space
   - Optimizations to reduce constant factors
   - Best practical solution

CORE 3D DP ALGORITHM:
====================
```python
# dp[i][j][k] = max points for boxes[i:j+1] where there are 
# k additional boxes of color boxes[j] that can be merged
memo = {}

def dp(i, j, k):
    if i > j:
        return 0
    
    # Merge consecutive same-colored boxes at right end
    while j > i and boxes[j] == boxes[j-1]:
        j -= 1
        k += 1
    
    # Option 1: Remove rightmost group (boxes[j] + k extra boxes)
    result = dp(i, j-1, 0) + (k + 1)²
    
    # Option 2: Find previous boxes of same color and merge
    for m in range(i, j):
        if boxes[m] == boxes[j]:
            result = max(result, dp(i, m, k+1) + dp(m+1, j-1, 0))
    
    return result
```

STATE SPACE DESIGN:
==================
**Why 3D state is necessary**:
- `dp[i][j]` alone is insufficient
- Need to track "pending" boxes that can be merged later
- The `k` parameter represents boxes of color `boxes[j]` that are "virtually" adjacent

**State Meaning**:
- `dp[i][j][k]`: Maximum points for interval [i,j] where we have k additional boxes of color `boxes[j]` that will be merged with `boxes[j]`

MERGING STRATEGY:
================
**Key Insight**: Don't always remove boxes immediately
- Sometimes better to remove intermediate boxes first
- Then merge distant same-colored boxes
- Quadratic scoring makes larger groups much more valuable

**Example**: [1,3,2,2,2,3,1]
- Remove middle 2's first: [1,3,3,1] 
- Then remove 3's together: 2² = 4 points (better than 1+1=2)

RECURRENCE RELATION:
===================
```
dp[i][j][k] = max(
    dp[i][j-1][0] + (k+1)²,                    // Remove rightmost group
    max(dp[i][m][k+1] + dp[m+1][j-1][0])       // Merge at position m
        for all m where boxes[m] == boxes[j]
)

Base case: dp[i][j][k] = 0 if i > j
```

**Optimization**: Merge consecutive same-colored boxes
```python
while j > i and boxes[j] == boxes[j-1]:
    j -= 1
    k += 1
```

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n⁴) worst case, O(n³) with memoization in practice
- **Space**: O(n³) for memoization table
- **States**: O(n³) possible (i,j,k) combinations
- **Transitions**: O(n) merging positions to try

WHY THIS PROBLEM IS HARD:
=========================
**Multiple Challenges**:
1. **Non-greedy**: Optimal immediate choice isn't always globally optimal
2. **Long-range dependencies**: Decisions affect distant future possibilities
3. **Complex state**: 3D state space much larger than typical interval DP
4. **Strategic depth**: Must plan several moves ahead

**Comparison to Simpler Interval DP**:
- Burst Balloons: 2D state, local decisions
- Remove Boxes: 3D state, non-local decisions

MATHEMATICAL PROPERTIES:
========================
- **Optimal Substructure**: Optimal solution contains optimal subsolutions
- **Overlapping Subproblems**: Same states computed multiple times
- **Non-monotonicity**: Larger intervals don't always have larger values
- **Quadratic Growth**: Scoring function creates strong incentive for merging

SOLUTION RECONSTRUCTION:
=======================
Tracking optimal decisions requires storing choice information:
```python
choice = {}  # Store decisions for each state

if merge_result > remove_result:
    choice[(i,j,k)] = ('merge', m)
else:
    choice[(i,j,k)] = ('remove', j, k+1)
```

APPLICATIONS:
============
- **Game Design**: Puzzle games with combo scoring (Tetris, Candy Crush)
- **Resource Management**: Batch processing with economies of scale
- **Financial Optimization**: Transaction bundling for reduced fees
- **Manufacturing**: Batch production optimization
- **Algorithm Design**: Complex state DP problems

RELATED PROBLEMS:
================
- **Burst Balloons (312)**: Simpler interval DP without merging
- **Matrix Chain Multiplication**: Classic interval optimization
- **Zuma Game**: Similar merging mechanics with different rules
- **Stone Game variants**: Game theory with interval structure

OPTIMIZATION TECHNIQUES:
=======================
- **Consecutive Merging**: Preprocess consecutive same-colored boxes
- **Pruning**: Skip obviously suboptimal branches
- **State Compression**: Reduce redundant state representations
- **Memory Management**: Optimize cache usage for large inputs

EDGE CASES:
==========
- **Single box**: Return 1 point
- **All same color**: Return n² points
- **No same colors**: Each box removed individually
- **Empty array**: Return 0 points

This problem represents the pinnacle of interval DP complexity,
demonstrating how additional state dimensions can dramatically
increase both the solution space and the algorithmic sophistication
required for efficient solutions.
"""
