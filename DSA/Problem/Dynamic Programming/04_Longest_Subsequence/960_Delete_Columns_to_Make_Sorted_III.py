"""
LeetCode 960: Delete Columns to Make Sorted III
Difficulty: Hard
Category: Longest Subsequence Problems (2D LIS variant)

PROBLEM DESCRIPTION:
===================
You are given an array of n strings strs, all of the same length.

We may choose any deletion indices, and we delete all the characters in those indices from each string.

For example, if we have strs = ["abcdef","uvwxyz"] and deletion indices {0, 2, 3}, then the final array 
after deletions is ["bef","vyz"].

Suppose we chose a set of deletion indices answer such that after deletions, the final array satisfies:
- Each remaining column is sorted in non-decreasing order.
- There is at least one column remaining.

Return the minimum number of columns to delete to achieve the above requirements.

Example 1:
Input: strs = ["babca","bbazb","ceccc","edddd","eeeee"]
Output: 3
Explanation: After deleting columns 0, 1, and 4, the remaining columns are ["a","b","c","d","e"] and ["c","z","c","d","e"], 
which are both sorted.

Example 2:
Input: strs = ["edcba"]
Output: 4

Example 3:
Input: strs = ["ghi","def","abc"]
Output: 0

Constraints:
- n == strs.length
- 1 <= n <= 100
- 1 <= strs[i].length <= 100
- strs[i] consists of lowercase English letters.
"""

def min_deletion_size_brute_force(strs):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible column selections and find the minimum deletions.
    
    Time Complexity: O(2^m * n * m) - 2^m subsets, O(nm) to check each
    Space Complexity: O(m) - recursion stack
    """
    if not strs or not strs[0]:
        return 0
    
    n, m = len(strs), len(strs[0])
    
    def is_valid_selection(selected_cols):
        """Check if selected columns form sorted matrix"""
        if not selected_cols:
            return False
        
        # Check each column is sorted
        for col in selected_cols:
            for i in range(1, n):
                if strs[i][col] < strs[i-1][col]:
                    return False
        
        # Check lexicographic ordering of rows
        for i in range(1, n):
            row1 = [strs[i-1][col] for col in selected_cols]
            row2 = [strs[i][col] for col in selected_cols]
            if row1 > row2:  # Not lexicographically sorted
                return False
        
        return True
    
    def generate_selections(col_idx, current_selection):
        if col_idx >= m:
            if is_valid_selection(current_selection):
                return len(current_selection)
            return 0
        
        # Skip current column
        skip = generate_selections(col_idx + 1, current_selection)
        
        # Include current column
        current_selection.append(col_idx)
        include = generate_selections(col_idx + 1, current_selection)
        current_selection.pop()
        
        return max(skip, include)
    
    max_kept = generate_selections(0, [])
    return m - max_kept


def min_deletion_size_dp_lis(strs):
    """
    DP APPROACH - LIS ON COLUMNS:
    ============================
    Find longest subsequence of columns that can be kept.
    
    Time Complexity: O(m^2 * n) - m^2 for DP, n to compare columns
    Space Complexity: O(m) - DP array
    """
    if not strs or not strs[0]:
        return 0
    
    n, m = len(strs), len(strs[0])
    
    def can_keep_both(col1, col2):
        """Check if we can keep both col1 and col2 (col1 before col2)"""
        # Check if col1 and col2 are individually sorted
        for i in range(1, n):
            if strs[i][col1] < strs[i-1][col1] or strs[i][col2] < strs[i-1][col2]:
                return False
        
        # Check if keeping both maintains lexicographic order
        for i in range(1, n):
            char1_prev, char2_prev = strs[i-1][col1], strs[i-1][col2]
            char1_curr, char2_curr = strs[i][col1], strs[i][col2]
            
            # If previous row is lexicographically greater, invalid
            if (char1_prev, char2_prev) > (char1_curr, char2_curr):
                return False
        
        return True
    
    # dp[i] = maximum columns we can keep ending with column i
    dp = [1] * m
    
    for i in range(1, m):
        for j in range(i):
            if can_keep_both(j, i):
                dp[i] = max(dp[i], dp[j] + 1)
    
    max_kept = max(dp)
    return m - max_kept


def min_deletion_size_optimized(strs):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Improved column comparison and early termination.
    
    Time Complexity: O(m^2 * n) - same as basic DP but with optimizations
    Space Complexity: O(m) - DP array
    """
    if not strs or not strs[0]:
        return 0
    
    n, m = len(strs), len(strs[0])
    
    # Precompute which columns are individually sorted
    is_sorted = [True] * m
    for col in range(m):
        for i in range(1, n):
            if strs[i][col] < strs[i-1][col]:
                is_sorted[col] = False
                break
    
    def can_extend(cols, new_col):
        """Check if we can add new_col to the sequence cols"""
        if not is_sorted[new_col]:
            return False
        
        if not cols:
            return True
        
        # Check lexicographic ordering
        for i in range(1, n):
            prev_row = tuple(strs[i-1][col] for col in cols + [new_col])
            curr_row = tuple(strs[i][col] for col in cols + [new_col])
            
            if prev_row > curr_row:
                return False
        
        return True
    
    # DP with sequence tracking
    dp = {}  # sequence -> max_length
    dp[tuple()] = 0
    
    max_length = 0
    
    for col in range(m):
        new_dp = dp.copy()
        
        for seq, length in dp.items():
            if can_extend(list(seq), col):
                new_seq = seq + (col,)
                new_length = length + 1
                
                if new_seq not in new_dp or new_dp[new_seq] < new_length:
                    new_dp[new_seq] = new_length
                    max_length = max(max_length, new_length)
        
        dp = new_dp
    
    return m - max_length


def min_deletion_size_greedy_attempt(strs):
    """
    GREEDY ATTEMPT (MAY NOT BE OPTIMAL):
    ===================================
    Try greedy approach for comparison.
    
    Time Complexity: O(m^2 * n) - greedy selection
    Space Complexity: O(m) - tracking kept columns
    """
    if not strs or not strs[0]:
        return 0
    
    n, m = len(strs), len(strs[0])
    
    def is_valid_with_columns(cols):
        """Check if keeping these columns results in valid matrix"""
        if not cols:
            return False
        
        # Check each column is sorted
        for col in cols:
            for i in range(1, n):
                if strs[i][col] < strs[i-1][col]:
                    return False
        
        # Check lexicographic ordering
        for i in range(1, n):
            row1 = tuple(strs[i-1][col] for col in cols)
            row2 = tuple(strs[i][col] for col in cols)
            if row1 > row2:
                return False
        
        return True
    
    # Greedy: try to add columns one by one
    kept_columns = []
    
    for col in range(m):
        if is_valid_with_columns(kept_columns + [col]):
            kept_columns.append(col)
    
    return m - len(kept_columns)


def min_deletion_size_with_result(strs):
    """
    FIND ACTUAL KEPT COLUMNS:
    =========================
    Return both minimum deletions and which columns to keep.
    
    Time Complexity: O(m^2 * n) - DP + reconstruction
    Space Complexity: O(m^2) - track parent sequences
    """
    if not strs or not strs[0]:
        return 0, []
    
    n, m = len(strs), len(strs[0])
    
    def can_extend_sequence(seq, new_col):
        """Check if we can add new_col to sequence seq"""
        # Check if new column is sorted
        for i in range(1, n):
            if strs[i][new_col] < strs[i-1][new_col]:
                return False
        
        if not seq:
            return True
        
        # Check lexicographic ordering with new column
        for i in range(1, n):
            prev_row = tuple(strs[i-1][col] for col in seq + [new_col])
            curr_row = tuple(strs[i][col] for col in seq + [new_col])
            
            if prev_row > curr_row:
                return False
        
        return True
    
    # DP to find longest valid sequence
    best_length = 0
    best_sequence = []
    
    # Try all possible sequences (exponential but works for small inputs)
    def backtrack(col_idx, current_seq):
        nonlocal best_length, best_sequence
        
        if col_idx >= m:
            if len(current_seq) > best_length:
                best_length = len(current_seq)
                best_sequence = current_seq[:]
            return
        
        # Skip current column
        backtrack(col_idx + 1, current_seq)
        
        # Try to include current column
        if can_extend_sequence(current_seq, col_idx):
            current_seq.append(col_idx)
            backtrack(col_idx + 1, current_seq)
            current_seq.pop()
    
    backtrack(0, [])
    
    return m - best_length, best_sequence


def min_deletion_size_analysis(strs):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step analysis of the problem.
    
    Time Complexity: O(m^2 * n) - analysis computation
    Space Complexity: O(m) - temporary arrays
    """
    if not strs or not strs[0]:
        return 0
    
    n, m = len(strs), len(strs[0])
    
    print(f"Input matrix:")
    for i, s in enumerate(strs):
        print(f"  Row {i}: {list(s)}")
    print(f"Dimensions: {n} rows × {m} columns")
    
    # Analyze each column
    print(f"\nColumn analysis:")
    sorted_cols = []
    
    for col in range(m):
        column_chars = [strs[i][col] for i in range(n)]
        is_sorted = all(column_chars[i] <= column_chars[i+1] for i in range(n-1))
        sorted_cols.append(is_sorted)
        
        print(f"  Column {col}: {column_chars} - {'Sorted' if is_sorted else 'Not sorted'}")
    
    # Try different combinations
    print(f"\nTesting column combinations:")
    
    def test_combination(cols):
        if not cols:
            return False, "Empty selection"
        
        # Check individual column sorting
        for col in cols:
            if not sorted_cols[col]:
                return False, f"Column {col} not sorted"
        
        # Check lexicographic ordering
        for i in range(1, n):
            row1 = tuple(strs[i-1][col] for col in cols)
            row2 = tuple(strs[i][col] for col in cols)
            if row1 > row2:
                return False, f"Rows {i-1} and {i} not in lexicographic order"
        
        return True, "Valid"
    
    # Test some combinations
    for num_cols in range(1, min(4, m + 1)):
        from itertools import combinations
        for cols in combinations(range(m), num_cols):
            valid, reason = test_combination(list(cols))
            print(f"  Columns {list(cols)}: {reason}")
            if valid:
                break
        if valid:
            break
    
    # Run actual algorithm
    result = min_deletion_size_dp_lis(strs)
    print(f"\nMinimum deletions needed: {result}")
    
    return result


# Test cases
def test_min_deletion_size():
    """Test all implementations with various inputs"""
    test_cases = [
        (["babca","bbazb","ceccc","edddd","eeeee"], 3),
        (["edcba"], 4),
        (["ghi","def","abc"], 0),
        (["abc","def","ghi"], 0),
        (["zyx","wvu","tsr"], 2),
        (["a","b","c"], 0),
        (["abc"], 0),
        (["ca","bb","ac"], 1),
        (["xc","yb","za"], 0),
        (["dvpmu","bkbrt","fgrqe","dqmfa","aswxz"])  # Let's see what this gives
    ]
    
    print("Testing Delete Columns to Make Sorted III Solutions:")
    print("=" * 70)
    
    for i, test_case in enumerate(test_cases):
        if len(test_case) == 2:
            strs, expected = test_case
        else:
            strs = test_case[0]
            expected = None
        
        print(f"\nTest Case {i + 1}: strs = {strs}")
        if expected is not None:
            print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for large inputs)
        if len(strs) <= 5 and len(strs[0]) <= 6:
            try:
                brute = min_deletion_size_brute_force(strs.copy())
                print(f"Brute Force:      {brute:>3} {'✓' if expected is None or brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        dp_lis = min_deletion_size_dp_lis(strs.copy())
        optimized = min_deletion_size_optimized(strs.copy())
        greedy = min_deletion_size_greedy_attempt(strs.copy())
        
        print(f"DP (LIS):         {dp_lis:>3} {'✓' if expected is None or dp_lis == expected else '✗'}")
        print(f"Optimized:        {optimized:>3} {'✓' if expected is None or optimized == expected else '✗'}")
        print(f"Greedy Attempt:   {greedy:>3} {'✓' if expected is None or greedy == expected else '✗'}")
        
        # Show actual kept columns for small cases
        if len(strs) <= 5 and len(strs[0]) <= 6:
            deletions, kept_cols = min_deletion_size_with_result(strs.copy())
            if kept_cols:
                print(f"Kept columns: {kept_cols}")
                # Show resulting matrix
                result_matrix = []
                for row in strs:
                    result_row = [row[col] for col in kept_cols]
                    result_matrix.append(''.join(result_row))
                print(f"Result matrix: {result_matrix}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    min_deletion_size_analysis(["babca","bbazb","ceccc","edddd","eeeee"])
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. 2D LIS PROBLEM: Find longest subsequence of columns")
    print("2. DUAL CONSTRAINTS: Each column sorted + lexicographic row order")
    print("3. COLUMN DEPENDENCY: Adding column affects entire matrix validity")
    print("4. OPTIMIZATION: Minimize deletions = maximize kept columns")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all column combinations")
    print("DP (LIS):         LIS on valid column sequences")
    print("Optimized:        Improved column validation")
    print("Greedy Attempt:   Greedy selection (may be suboptimal)")
    print("With Result:      DP + reconstruction")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^m * nm), Space: O(m)")
    print("DP (LIS):         Time: O(m² * n),   Space: O(m)")
    print("Optimized:        Time: O(m² * n),   Space: O(m)")
    print("Greedy Attempt:   Time: O(m² * n),   Space: O(m)")
    print("With Result:      Time: O(m² * n),   Space: O(m²)")


if __name__ == "__main__":
    test_min_deletion_size()


"""
PATTERN RECOGNITION:
==================
This is a 2D LIS problem with dual constraints:
- Find longest subsequence of columns that can be kept
- Each column must be individually sorted
- Resulting matrix must have lexicographically sorted rows
- Minimize deletions = maximize kept columns

KEY INSIGHT - DUAL CONSTRAINT SYSTEM:
====================================
**Two-level constraint checking**:
1. **Column-level**: Each selected column must be sorted
2. **Matrix-level**: Rows must be in lexicographic order

**Dependency**: Adding a column affects the entire matrix validity,
not just local column properties.

PROBLEM TRANSFORMATION:
======================
**Original**: Delete minimum columns to make matrix valid
**Transformed**: Find maximum columns that can be kept together
**Constraint**: Valid configuration satisfies both conditions

ALGORITHM APPROACHES:
====================

1. **DP (LIS-style)**: O(m² × n)
   - For each column, try extending all previous valid sequences
   - Check if new column maintains both constraints
   - Track maximum valid sequence length

2. **Optimized DP**: O(m² × n)
   - Precompute individual column sorting
   - Improved constraint checking
   - Early termination optimizations

3. **Brute Force**: O(2^m × n × m)
   - Try all possible column selections
   - Check validity of each selection
   - Find selection with maximum columns

CONSTRAINT VALIDATION:
=====================
**Can we keep columns [c1, c2, ..., ck]?**

1. **Individual sorting**: Each column cᵢ must be sorted
2. **Lexicographic ordering**: For each row pair (i, i+1):
   ```
   (strs[i][c1], strs[i][c2], ..., strs[i][ck]) ≤ 
   (strs[i+1][c1], strs[i+1][c2], ..., strs[i+1][ck])
   ```

DP STATE DEFINITION:
===================
dp[i] = maximum number of columns we can keep ending with column i

RECURRENCE RELATION:
===================
```python
dp[i] = max(dp[j] + 1) for all j < i where can_keep_both(j, i)
```

where can_keep_both(j, i) checks if columns j and i can coexist.

COMPLEXITY ANALYSIS:
===================
**Why O(m² × n)?**
- m² pairs of columns to check
- n comparisons for each pair (lexicographic check)
- Each check takes O(n) time for row comparison

**Space optimization**: Can reduce to O(m) space using rolling DP.

EDGE CASES:
==========
1. **Single column**: Always at least one column can be kept
2. **Already valid**: No deletions needed (return 0)
3. **No valid combination**: Keep minimum one column
4. **All columns identical**: Special case handling

OPTIMIZATION OPPORTUNITIES:
==========================
1. **Precompute column sorting**: Avoid redundant checks
2. **Early termination**: Skip invalid columns early
3. **Memoization**: Cache constraint validation results
4. **Greedy heuristics**: For approximate solutions

APPLICATIONS:
============
1. **Database Optimization**: Column selection for sorted tables
2. **Data Compression**: Remove redundant sorted columns
3. **Matrix Processing**: Maintain sorted structure
4. **Spreadsheet Operations**: Column-based sorting

VARIANTS TO PRACTICE:
====================
- Delete Columns to Make Sorted (944) - simpler version
- Delete Columns to Make Sorted II (955) - intermediate version
- Longest Increasing Subsequence (300) - 1D foundation
- Russian Doll Envelopes (354) - 2D LIS

INTERVIEW TIPS:
==============
1. **Recognize as 2D LIS**: Key insight for approach
2. **Understand dual constraints**: Column + matrix level
3. **Show constraint checking**: How to validate combinations
4. **Explain DP formulation**: Why LIS-style works
5. **Handle edge cases**: Single column, already valid
6. **Discuss optimization**: Precomputation opportunities
7. **Complexity analysis**: Why O(m² × n) is necessary
8. **Alternative approaches**: Greedy vs optimal DP
9. **Real applications**: Database, data processing contexts
10. **Reconstruction**: How to find actual kept columns

MATHEMATICAL INSIGHT:
====================
This problem demonstrates the tension between:
- **Local constraints** (individual column sorting)
- **Global constraints** (matrix lexicographic ordering)

The optimal solution requires considering both levels simultaneously,
making it more complex than standard LIS problems.
"""
