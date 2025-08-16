"""
LeetCode 72: Edit Distance (Levenshtein Distance)
Difficulty: Hard
Category: String DP

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

def min_distance_recursive(word1, word2):
    """
    RECURSIVE APPROACH:
    ==================
    Try all possible operations recursively.
    
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
            operations[0][j] = f"Insert '{word2[j-1]}' at position 0"
    
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
                    operations[i][j] = f"Insert '{word2[j-1]}' at position {i}"
    
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


def min_distance_variants():
    """
    EDIT DISTANCE VARIANTS:
    ======================
    Test different scenarios and variations.
    """
    from collections import defaultdict
    
    def edit_distance_with_costs(word1, word2, insert_cost, delete_cost, replace_cost):
        """Edit distance with different operation costs"""
        m, n = len(word1), len(word2)
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        
        # Base cases with custom costs
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + delete_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + insert_cost
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i][j-1] + insert_cost,
                        dp[i-1][j] + delete_cost,
                        dp[i-1][j-1] + replace_cost
                    )
        
        return dp[m][n]
    
    def constrained_edit_distance(word1, word2, forbidden_chars):
        """Edit distance where certain characters cannot be inserted"""
        m, n = len(word1), len(word2)
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        
        dp[0][0] = 0
        for i in range(1, m + 1):
            dp[i][0] = i  # Can always delete
        
        for j in range(1, n + 1):
            if word2[j-1] not in forbidden_chars:
                dp[0][j] = j  # Can insert if not forbidden
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Delete is always allowed
                    dp[i][j] = min(dp[i][j], dp[i-1][j] + 1)
                    
                    # Insert only if character is not forbidden
                    if word2[j-1] not in forbidden_chars:
                        dp[i][j] = min(dp[i][j], dp[i][j-1] + 1)
                    
                    # Replace only if target character is not forbidden
                    if word2[j-1] not in forbidden_chars:
                        dp[i][j] = min(dp[i][j], dp[i-1][j-1] + 1)
        
        return dp[m][n]
    
    # Test variants
    test_cases = [
        ("kitten", "sitting", "Basic edit distance"),
        ("horse", "ros", "Classic example"),
        ("intention", "execution", "Longer transformation")
    ]
    
    print("Edit Distance Variants Analysis:")
    print("=" * 50)
    
    for word1, word2, description in test_cases:
        print(f"\n{description}: '{word1}' -> '{word2}'")
        
        basic = min_distance_dp(word1, word2)
        print(f"  Basic edit distance: {basic}")
        
        # Custom costs
        weighted = edit_distance_with_costs(word1, word2, 1, 2, 3)  # Insert:1, Delete:2, Replace:3
        print(f"  Weighted (I:1,D:2,R:3): {weighted}")
        
        # Constrained
        forbidden = {'x', 'z'}
        constrained = constrained_edit_distance(word1, word2, forbidden)
        print(f"  Constrained (no x,z): {constrained}")


# Test cases
def test_edit_distance():
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
        ("a", "b", 1),
        ("abcdef", "azced", 3)
    ]
    
    print("Testing Edit Distance Solutions:")
    print("=" * 70)
    
    for i, (word1, word2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"word1='{word1}', word2='{word2}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip expensive ones for long strings)
        if len(word1) + len(word2) <= 10:
            try:
                recursive = min_distance_recursive(word1, word2)
                print(f"Recursive:        {recursive:>3} {'✓' if recursive == expected else '✗'}")
            except:
                print(f"Recursive:        Timeout")
        
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
                for op in operations[:3]:  # Show first 3 operations
                    print(f"  • {op}")
                if len(operations) > 3:
                    print(f"  ... and {len(operations) - 3} more")
    
    # Variants testing
    print(f"\n" + "=" * 70)
    min_distance_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. THREE OPERATIONS: Insert, Delete, Replace at each step")
    print("2. OPTIMAL SUBSTRUCTURE: Optimal solution contains optimal subsolutions")
    print("3. CHARACTER MATCHING: No cost when characters are the same")
    print("4. SPACE OPTIMIZATION: Can reduce to O(min(m,n)) space")
    print("5. OPERATION TRACKING: Can reconstruct actual transformation sequence")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Spell Checkers: Find closest dictionary words")
    print("• DNA Analysis: Sequence alignment in bioinformatics")
    print("• Version Control: Computing file differences")
    print("• Machine Translation: Text similarity measurement")
    print("• Fuzzy Search: Approximate string matching")


if __name__ == "__main__":
    test_edit_distance()


"""
EDIT DISTANCE - THE FOUNDATIONAL STRING DP PROBLEM:
==================================================

Edit Distance (Levenshtein Distance) is the cornerstone of string dynamic programming:
- Foundation for sequence alignment in bioinformatics
- Core algorithm for spell checkers and diff tools
- Prototype for transformation optimization problems
- Demonstrates classic 2D DP with optimal substructure

KEY INSIGHTS:
============
1. **THREE OPERATIONS**: Insert, Delete, Replace - covers all possible transformations
2. **CHARACTER MATCHING**: When characters match, no operation needed
3. **OPTIMAL SUBSTRUCTURE**: Optimal alignment contains optimal sub-alignments
4. **SPACE OPTIMIZATION**: Only need previous row for computation

RECURRENCE RELATION:
===================
```
if word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]  # No operation needed
else:
    dp[i][j] = 1 + min(
        dp[i-1][j],     # Delete
        dp[i][j-1],     # Insert  
        dp[i-1][j-1]    # Replace
    )
```

OPERATION SEMANTICS:
===================
- **Delete**: Remove character from word1, move i pointer
- **Insert**: Add character to word1, move j pointer
- **Replace**: Change character in word1, move both pointers

APPLICATIONS:
============
- **Bioinformatics**: DNA/protein sequence alignment
- **Spell Checking**: Finding closest dictionary words
- **Version Control**: Computing file differences (diff)
- **Machine Translation**: Measuring text similarity
- **Fuzzy Search**: Approximate string matching
- **Plagiarism Detection**: Text comparison

VARIANTS:
========
- **Weighted Operations**: Different costs for insert/delete/replace
- **Constrained Operations**: Forbidden characters or operations
- **Alignment with Gaps**: Biological sequence alignment
- **Local vs Global**: Substring vs full string alignment

COMPLEXITY:
==========
- **Time**: O(m×n) - optimal for this problem
- **Space**: O(m×n) standard, O(min(m,n)) optimized
- **Operations**: O(m+n) to reconstruct transformation sequence

This implementation provides the foundation for understanding all string DP problems.
The techniques learned here apply to LCS, palindromes, pattern matching, and more.
"""
