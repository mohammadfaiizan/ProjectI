"""
LeetCode 72: Edit Distance
Difficulty: Hard
Category: Grid/Matrix DP / String DP

PROBLEM DESCRIPTION:
===================
Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:
- Insert a character
- Delete a character
- Replace a character

Example 1:
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation: 
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation: 
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')

Constraints:
- 0 <= word1.length, word2.length <= 500
- word1 and word2 consist of lowercase English letters.
"""

def min_distance_brute_force(word1, word2):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible sequences of operations.
    
    Time Complexity: O(3^max(m,n)) - three operations at each step
    Space Complexity: O(max(m,n)) - recursion depth
    """
    def dfs(i, j):
        # Base cases
        if i >= len(word1):
            return len(word2) - j  # Insert remaining characters from word2
        
        if j >= len(word2):
            return len(word1) - i  # Delete remaining characters from word1
        
        # If characters match, no operation needed
        if word1[i] == word2[j]:
            return dfs(i + 1, j + 1)
        
        # Try all three operations
        insert = 1 + dfs(i, j + 1)      # Insert word2[j], advance j
        delete = 1 + dfs(i + 1, j)      # Delete word1[i], advance i
        replace = 1 + dfs(i + 1, j + 1) # Replace word1[i] with word2[j]
        
        return min(insert, delete, replace)
    
    return dfs(0, 0)


def min_distance_memoization(word1, word2):
    """
    MEMOIZATION APPROACH:
    ====================
    Use memoization to avoid recomputing same subproblems.
    
    Time Complexity: O(m*n) - each state computed once
    Space Complexity: O(m*n) - memoization table
    """
    memo = {}
    
    def dfs(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        # Base cases
        if i >= len(word1):
            result = len(word2) - j
        elif j >= len(word2):
            result = len(word1) - i
        elif word1[i] == word2[j]:
            result = dfs(i + 1, j + 1)
        else:
            # Try all three operations
            insert = 1 + dfs(i, j + 1)
            delete = 1 + dfs(i + 1, j)
            replace = 1 + dfs(i + 1, j + 1)
            result = min(insert, delete, replace)
        
        memo[(i, j)] = result
        return result
    
    return dfs(0, 0)


def min_distance_dp(word1, word2):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Build solution bottom-up using 2D DP table.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(m*n) - DP table
    """
    m, n = len(word1), len(word2)
    
    # dp[i][j] = min operations to convert word1[0:i] to word2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: converting empty string to word2[0:j] or word1[0:i] to empty string
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters from word1[0:i]
    
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters from word2[0:j]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                # Characters match, no operation needed
                dp[i][j] = dp[i-1][j-1]
            else:
                # Try all three operations and take minimum
                insert = dp[i][j-1] + 1      # Insert word2[j-1]
                delete = dp[i-1][j] + 1      # Delete word1[i-1]
                replace = dp[i-1][j-1] + 1   # Replace word1[i-1] with word2[j-1]
                
                dp[i][j] = min(insert, delete, replace)
    
    return dp[m][n]


def min_distance_space_optimized(word1, word2):
    """
    SPACE OPTIMIZED APPROACH:
    =========================
    Use only O(min(m,n)) space by processing row by row.
    
    Time Complexity: O(m*n) - process each cell once
    Space Complexity: O(min(m,n)) - single row array
    """
    # Ensure word1 is the shorter string for space optimization
    if len(word1) > len(word2):
        word1, word2 = word2, word1
    
    m, n = len(word1), len(word2)
    
    # Use single array to represent previous row
    prev = list(range(m + 1))  # Base case: word1[0:i] to empty string
    
    for j in range(1, n + 1):
        curr = [j]  # Base case: empty string to word2[0:j]
        
        for i in range(1, m + 1):
            if word1[i-1] == word2[j-1]:
                curr.append(prev[i-1])  # Characters match
            else:
                insert = curr[i-1] + 1    # Insert
                delete = prev[i] + 1      # Delete
                replace = prev[i-1] + 1   # Replace
                curr.append(min(insert, delete, replace))
        
        prev = curr
    
    return prev[m]


def min_distance_with_operations(word1, word2):
    """
    FIND ACTUAL OPERATION SEQUENCE:
    ===============================
    Return minimum operations and the actual sequence of operations.
    
    Time Complexity: O(m*n) - DP + operation reconstruction
    Space Complexity: O(m*n) - DP table + operation tracking
    """
    m, n = len(word1), len(word2)
    
    # DP table and operation tracking
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    operations = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            operations[i][0] = f"Delete '{word1[i-1]}' at position {i-1}"
    
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            operations[0][j] = f"Insert '{word2[j-1]}' at position {j-1}"
    
    # Fill DP table with operation tracking
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                operations[i][j] = "Match"
            else:
                # Calculate costs for each operation
                insert_cost = dp[i][j-1] + 1
                delete_cost = dp[i-1][j] + 1
                replace_cost = dp[i-1][j-1] + 1
                
                min_cost = min(insert_cost, delete_cost, replace_cost)
                dp[i][j] = min_cost
                
                # Track which operation was chosen
                if min_cost == replace_cost:
                    operations[i][j] = f"Replace '{word1[i-1]}' with '{word2[j-1]}' at position {i-1}"
                elif min_cost == delete_cost:
                    operations[i][j] = f"Delete '{word1[i-1]}' at position {i-1}"
                else:  # insert_cost
                    operations[i][j] = f"Insert '{word2[j-1]}' at position {i-1}"
    
    # Reconstruct operation sequence
    operation_sequence = []
    i, j = m, n
    
    while i > 0 or j > 0:
        op = operations[i][j]
        if op and op != "Match":
            operation_sequence.append(op)
        
        if i > 0 and j > 0 and word1[i-1] == word2[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            i -= 1
        else:
            j -= 1
    
    operation_sequence.reverse()
    
    return dp[m][n], operation_sequence


def min_distance_analysis(word1, word2):
    """
    DETAILED ANALYSIS:
    =================
    Show step-by-step DP computation and operation analysis.
    
    Time Complexity: O(m*n) - analysis computation
    Space Complexity: O(m*n) - temporary tables
    """
    m, n = len(word1), len(word2)
    
    print(f"Input strings:")
    print(f"  word1 = '{word1}' (length {m})")
    print(f"  word2 = '{word2}' (length {n})")
    
    print(f"\nDP table construction:")
    print(f"  dp[i][j] = min operations to convert word1[0:i] to word2[0:j]")
    
    # Create DP table for visualization
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    print(f"\nBase cases:")
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            print(f"  dp[{i}][0] = {i} (delete all {i} characters from word1)")
    
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            print(f"  dp[0][{j}] = {j} (insert all {j} characters from word2)")
    
    # Fill DP table
    print(f"\nFilling DP table:")
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            char1, char2 = word1[i-1], word2[j-1]
            
            if char1 == char2:
                dp[i][j] = dp[i-1][j-1]
                print(f"  dp[{i}][{j}]: '{char1}' == '{char2}' → dp[{i-1}][{j-1}] = {dp[i][j]}")
            else:
                insert = dp[i][j-1] + 1
                delete = dp[i-1][j] + 1
                replace = dp[i-1][j-1] + 1
                
                dp[i][j] = min(insert, delete, replace)
                
                print(f"  dp[{i}][{j}]: '{char1}' != '{char2}'")
                print(f"    Insert: {dp[i][j-1]} + 1 = {insert}")
                print(f"    Delete: {dp[i-1][j]} + 1 = {delete}")
                print(f"    Replace: {dp[i-1][j-1]} + 1 = {replace}")
                print(f"    Min: {dp[i][j]}")
    
    print(f"\nFinal DP table:")
    # Print column headers
    print("     ", end="")
    print("   ε", end="")
    for c in word2:
        print(f"   {c}", end="")
    print()
    
    # Print rows
    for i in range(m + 1):
        if i == 0:
            print("  ε: ", end="")
        else:
            print(f"  {word1[i-1]}: ", end="")
        
        for j in range(n + 1):
            print(f"{dp[i][j]:4}", end="")
        print()
    
    result = dp[m][n]
    print(f"\nMinimum edit distance: {result}")
    
    # Show actual operations
    min_ops, operations = min_distance_with_operations(word1, word2)
    if operations:
        print(f"\nOperation sequence:")
        for i, op in enumerate(operations):
            print(f"  {i+1}. {op}")
    
    return result


def min_distance_variants():
    """
    TEST PROBLEM VARIANTS:
    =====================
    Test different scenarios and edge cases.
    """
    test_cases = [
        # Basic cases
        ("horse", "ros", 3, "Classic example"),
        ("intention", "execution", 5, "Longer strings"),
        
        # Edge cases
        ("", "", 0, "Both empty"),
        ("", "abc", 3, "First empty"),
        ("abc", "", 3, "Second empty"),
        ("a", "a", 0, "Single character match"),
        ("a", "b", 1, "Single character replace"),
        
        # Special patterns
        ("abc", "abc", 0, "Identical strings"),
        ("abc", "cba", 2, "Reversed strings"),
        ("kitten", "sitting", 3, "Common example"),
        ("saturday", "sunday", 3, "Similar strings"),
        ("ab", "ba", 2, "Simple swap"),
        ("abc", "def", 3, "No common characters")
    ]
    
    print("Testing Edit Distance Variants:")
    print("=" * 60)
    
    for i, (word1, word2, expected, description) in enumerate(test_cases):
        print(f"\nTest {i+1}: {description}")
        print(f"'{word1}' → '{word2}'")
        print(f"Expected: {expected}")
        
        result = min_distance_dp(word1, word2)
        print(f"Result: {result} {'✓' if result == expected else '✗'}")
        
        if len(word1) <= 5 and len(word2) <= 5:
            _, operations = min_distance_with_operations(word1, word2)
            if operations:
                print(f"Operations: {len(operations)}")
                for op in operations[:3]:  # Show first 3 operations
                    print(f"  • {op}")
                if len(operations) > 3:
                    print(f"  ... and {len(operations) - 3} more")


# Test cases
def test_min_distance():
    """Test all implementations with various inputs"""
    test_cases = [
        ("horse", "ros", 3),
        ("intention", "execution", 5),
        ("", "", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("abc", "abc", 0),
        ("kitten", "sitting", 3),
        ("saturday", "sunday", 3),
        ("ab", "ba", 2),
        ("a", "b", 1)
    ]
    
    print("Testing Edit Distance Solutions:")
    print("=" * 70)
    
    for i, (word1, word2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"word1='{word1}', word2='{word2}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(word1) + len(word2) <= 8:
            try:
                brute = min_distance_brute_force(word1, word2)
                print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memo = min_distance_memoization(word1, word2)
        dp_result = min_distance_dp(word1, word2)
        space_opt = min_distance_space_optimized(word1, word2)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"DP (2D):          {dp_result:>3} {'✓' if dp_result == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        
        # Show operations for small cases
        if len(word1) <= 6 and len(word2) <= 6:
            min_ops, operations = min_distance_with_operations(word1, word2)
            print(f"Operations: {min_ops}")
            if operations:
                for op in operations[:2]:  # Show first 2 operations
                    print(f"  • {op}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    min_distance_analysis("horse", "ros")
    
    # Variants testing
    print(f"\n" + "=" * 70)
    min_distance_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. THREE OPERATIONS: Insert, Delete, Replace at each step")
    print("2. OPTIMAL SUBSTRUCTURE: Optimal solution contains optimal subsolutions")
    print("3. OVERLAPPING SUBPROBLEMS: Same (i,j) states reached multiple ways")
    print("4. CHARACTER MATCHING: No cost when characters are the same")
    print("5. SPACE OPTIMIZATION: Can reduce to O(min(m,n)) space")
    
    print("\n" + "=" * 70)
    print("Algorithm Comparison:")
    print("Brute Force:      Try all operation sequences")
    print("Memoization:      Top-down with caching")
    print("DP (2D):          Bottom-up with 2D table")
    print("Space Optimized:  Row-by-row processing")
    print("With Operations:  DP + operation reconstruction")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(3^max(m,n)), Space: O(max(m,n))")
    print("Memoization:      Time: O(m*n),        Space: O(m*n)")
    print("DP (2D):          Time: O(m*n),        Space: O(m*n)")
    print("Space Optimized:  Time: O(m*n),        Space: O(min(m,n))")
    print("With Operations:  Time: O(m*n),        Space: O(m*n)")


if __name__ == "__main__":
    test_min_distance()


"""
PATTERN RECOGNITION:
==================
Edit Distance (Levenshtein Distance) is the quintessential string DP problem:
- Foundation for sequence alignment in bioinformatics
- Core algorithm for spell checkers and diff tools
- Prototype for many transformation optimization problems
- Demonstrates classic 2D DP with three-way recurrence

KEY INSIGHT - OPERATION SEMANTICS:
=================================
**Three Basic Operations**:
1. **Insert**: Add character from target string
2. **Delete**: Remove character from source string  
3. **Replace**: Change character in source to match target

**State Transition Logic**:
- If characters match: No operation needed, advance both pointers
- If characters differ: Try all three operations, take minimum cost
- Each operation has cost 1 (can be generalized to different costs)

ALGORITHM APPROACHES:
====================

1. **2D DP (Standard)**: O(m×n) time, O(m×n) space
   - Most systematic and easy to understand
   - Allows operation reconstruction
   - Foundation for variants and extensions

2. **Space Optimized**: O(m×n) time, O(min(m,n)) space
   - Critical optimization for large strings
   - Only need previous row to compute current row
   - Standard interview follow-up question

3. **Memoization**: O(m×n) time, O(m×n) space
   - Top-down approach with natural recursion
   - Good for understanding the problem structure

4. **Brute Force**: O(3^max(m,n)) time
   - Try all possible operation sequences
   - Exponential complexity, only for tiny inputs

DP STATE DEFINITION:
===================
```
dp[i][j] = minimum operations to convert word1[0:i] to word2[0:j]
```

**Goal**: Find dp[m][n] where m = len(word1), n = len(word2)

**Base Cases**:
- dp[i][0] = i (delete all i characters from word1)
- dp[0][j] = j (insert all j characters to reach word2)

RECURRENCE RELATION:
===================
```
if word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]  # No operation needed
else:
    dp[i][j] = 1 + min(
        dp[i-1][j],     # Delete word1[i-1]
        dp[i][j-1],     # Insert word2[j-1]  
        dp[i-1][j-1]    # Replace word1[i-1] with word2[j-1]
    )
```

**Operation Interpretation**:
- **Delete**: Move from (i-1,j) to (i,j) - remove word1[i-1]
- **Insert**: Move from (i,j-1) to (i,j) - add word2[j-1]
- **Replace**: Move from (i-1,j-1) to (i,j) - change word1[i-1] to word2[j-1]

SPACE OPTIMIZATION:
==================
**Key Observation**: Only need previous row to compute current row

```python
prev = list(range(m + 1))  # Base case for first row

for j in range(1, n + 1):
    curr = [j]  # Base case for first column
    
    for i in range(1, m + 1):
        if word1[i-1] == word2[j-1]:
            curr.append(prev[i-1])
        else:
            insert = curr[i-1] + 1
            delete = prev[i] + 1
            replace = prev[i-1] + 1
            curr.append(min(insert, delete, replace))
    
    prev = curr
```

**Further Optimization**: Ensure the shorter string is used for the rolling dimension.

OPERATION RECONSTRUCTION:
========================
To find actual sequence of operations:
1. **Track choices** during DP computation
2. **Backtrack** from dp[m][n] to dp[0][0]
3. **Record** which operation was taken at each step

```python
# During DP, track the operation chosen
if min_cost == replace_cost:
    operation[i][j] = "Replace"
elif min_cost == delete_cost:
    operation[i][j] = "Delete"
else:
    operation[i][j] = "Insert"

# Backtrack to reconstruct sequence
operations = []
i, j = m, n
while i > 0 or j > 0:
    # Determine which operation was used
    # Add to operations list
    # Update i, j accordingly
```

APPLICATIONS:
============
1. **Bioinformatics**: DNA/protein sequence alignment
2. **Spell Checkers**: Finding closest dictionary words
3. **Version Control**: Computing file differences (diff)
4. **Plagiarism Detection**: Measuring text similarity
5. **Machine Translation**: Alignment of parallel corpora
6. **Database Fuzzy Search**: Finding approximate matches

VARIANTS TO PRACTICE:
====================
- One Edit Distance (161) - check if exactly one edit away
- Delete Operation for Two Strings (583) - only delete operations
- Minimum ASCII Delete Sum (712) - weighted operations
- Edit Distance with Different Costs - non-uniform operation costs

INTERVIEW TIPS:
==============
1. **Understand three operations**: Clearly explain insert/delete/replace
2. **Show recurrence logic**: Why we take min of three possibilities
3. **Handle base cases**: Empty string conversions
4. **Space optimization**: Critical follow-up question
5. **Operation reconstruction**: How to find actual edit sequence
6. **Real applications**: Bioinformatics, spell checking, diff tools
7. **Edge cases**: Empty strings, identical strings, single characters
8. **Complexity analysis**: Why O(mn) time is necessary
9. **Alternative formulations**: Different operation costs
10. **Related problems**: LCS relationship, sequence alignment

MATHEMATICAL PROPERTIES:
========================
- **Symmetric**: distance(A,B) = distance(B,A) with same operation costs
- **Triangle inequality**: distance(A,C) ≤ distance(A,B) + distance(B,C)
- **Non-negative**: distance(A,B) ≥ 0, equals 0 iff A = B
- **Metric space**: Forms a true metric with standard properties

OPTIMIZATION OPPORTUNITIES:
==========================
1. **Early termination**: If remaining characters exceed current minimum
2. **Diagonal optimization**: Use O(min(m,n)) space with diagonal processing
3. **Character frequency**: Pre-filter obviously impossible cases
4. **Suffix/prefix matching**: Skip identical prefixes and suffixes

MATHEMATICAL INSIGHT:
====================
Edit Distance represents the **minimum cost path** in a **2D lattice**:
- Each cell (i,j) represents a **state** in the transformation process
- Each **edge** represents an **operation** with associated cost
- **Optimal substructure** ensures DP finds the globally optimal solution

The problem elegantly demonstrates how **local optimization** (three choices at each step) 
leads to **global optimization** (minimum total edit distance).
"""
