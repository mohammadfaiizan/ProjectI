"""
LeetCode 72: Edit Distance (Levenshtein Distance)
Difficulty: Hard
Category: String DP

PROBLEM DESCRIPTION:
===================
Given two strings word1 and word2, return the minimum number of operations required 
to convert word1 to word2.

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

def edit_distance_bruteforce(word1, word2):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible edit operations recursively.
    
    Time Complexity: O(3^max(m,n)) - three choices at each step
    Space Complexity: O(max(m,n)) - recursion stack depth
    """
    def min_edit(i, j):
        # Base cases
        if i == len(word1):
            return len(word2) - j  # Insert remaining characters from word2
        if j == len(word2):
            return len(word1) - i  # Delete remaining characters from word1
        
        # If characters match, no operation needed
        if word1[i] == word2[j]:
            return min_edit(i + 1, j + 1)
        
        # Try all three operations
        insert_op = 1 + min_edit(i, j + 1)      # Insert character from word2
        delete_op = 1 + min_edit(i + 1, j)      # Delete character from word1
        replace_op = 1 + min_edit(i + 1, j + 1) # Replace character in word1
        
        return min(insert_op, delete_op, replace_op)
    
    return min_edit(0, 0)


def edit_distance_memoization(word1, word2):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(m * n) - each subproblem calculated once
    Space Complexity: O(m * n) - memoization table + recursion stack
    """
    memo = {}
    
    def min_edit(i, j):
        if i == len(word1):
            return len(word2) - j
        if j == len(word2):
            return len(word1) - i
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        if word1[i] == word2[j]:
            result = min_edit(i + 1, j + 1)
        else:
            insert_op = 1 + min_edit(i, j + 1)
            delete_op = 1 + min_edit(i + 1, j)
            replace_op = 1 + min_edit(i + 1, j + 1)
            result = min(insert_op, delete_op, replace_op)
        
        memo[(i, j)] = result
        return result
    
    return min_edit(0, 0)


def edit_distance_tabulation(word1, word2):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using 2D DP table.
    dp[i][j] = minimum edit distance between word1[0:i] and word2[0:j]
    
    Time Complexity: O(m * n) - fill entire DP table
    Space Complexity: O(m * n) - 2D DP table
    """
    m, n = len(word1), len(word2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    # Converting empty string to word2[0:j] requires j insertions
    for j in range(n + 1):
        dp[0][j] = j
    
    # Converting word1[0:i] to empty string requires i deletions
    for i in range(m + 1):
        dp[i][0] = i
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                # Characters match, no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Take minimum of three operations
                insert = dp[i][j - 1] + 1      # Insert
                delete = dp[i - 1][j] + 1      # Delete
                replace = dp[i - 1][j - 1] + 1 # Replace
                dp[i][j] = min(insert, delete, replace)
    
    return dp[m][n]


def edit_distance_space_optimized(word1, word2):
    """
    SPACE OPTIMIZED DYNAMIC PROGRAMMING:
    ===================================
    Since we only need previous row, use 1D array instead of 2D.
    
    Time Complexity: O(m * n) - same number of operations
    Space Complexity: O(min(m, n)) - use smaller dimension for space
    """
    # Ensure word2 is the shorter string for space optimization
    if len(word1) < len(word2):
        word1, word2 = word2, word1
    
    m, n = len(word1), len(word2)
    
    # Use array of size n+1
    prev = list(range(n + 1))  # Previous row
    curr = [0] * (n + 1)       # Current row
    
    for i in range(1, m + 1):
        curr[0] = i  # First column: delete all characters from word1[0:i]
        
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                insert = curr[j - 1] + 1    # Insert
                delete = prev[j] + 1        # Delete
                replace = prev[j - 1] + 1   # Replace
                curr[j] = min(insert, delete, replace)
        
        # Swap arrays
        prev, curr = curr, prev
    
    return prev[n]


def edit_distance_one_row(word1, word2):
    """
    FURTHER SPACE OPTIMIZED - SINGLE ROW:
    ====================================
    Use only one row and update in place with careful ordering.
    
    Time Complexity: O(m * n) - same operations
    Space Complexity: O(min(m, n)) - single array
    """
    if len(word1) < len(word2):
        word1, word2 = word2, word1
    
    m, n = len(word1), len(word2)
    dp = list(range(n + 1))
    
    for i in range(1, m + 1):
        prev_diag = dp[0]  # Store dp[i-1][j-1]
        dp[0] = i          # dp[i][0] = i
        
        for j in range(1, n + 1):
            temp = dp[j]   # Store dp[i-1][j] before updating
            
            if word1[i - 1] == word2[j - 1]:
                dp[j] = prev_diag
            else:
                insert = dp[j - 1] + 1   # dp[i][j-1]
                delete = dp[j] + 1       # dp[i-1][j]
                replace = prev_diag + 1  # dp[i-1][j-1]
                dp[j] = min(insert, delete, replace)
            
            prev_diag = temp
    
    return dp[n]


def edit_distance_with_operations(word1, word2):
    """
    DP WITH OPERATION TRACKING:
    ==========================
    Track the actual operations performed to transform word1 to word2.
    
    Time Complexity: O(m * n) - DP + operation reconstruction
    Space Complexity: O(m * n) - DP table + operation storage
    """
    m, n = len(word1), len(word2)
    
    # DP table and operation tracking
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            ops[0][j] = f"Insert '{word2[j-1]}'"
    
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            ops[i][0] = f"Delete '{word1[i-1]}'"
    
    # Fill DP table with operation tracking
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = "Match"
            else:
                insert_cost = dp[i][j - 1] + 1
                delete_cost = dp[i - 1][j] + 1
                replace_cost = dp[i - 1][j - 1] + 1
                
                min_cost = min(insert_cost, delete_cost, replace_cost)
                dp[i][j] = min_cost
                
                if min_cost == insert_cost:
                    ops[i][j] = f"Insert '{word2[j-1]}'"
                elif min_cost == delete_cost:
                    ops[i][j] = f"Delete '{word1[i-1]}'"
                else:
                    ops[i][j] = f"Replace '{word1[i-1]}' with '{word2[j-1]}'"
    
    # Reconstruct operations
    operations = []
    i, j = m, n
    
    while i > 0 or j > 0:
        op = ops[i][j]
        if op and op != "Match":
            operations.append(op)
        
        if i > 0 and j > 0 and word1[i - 1] == word2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
    
    operations.reverse()
    return dp[m][n], operations


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
        ("cat", "bat", 1),
        ("saturday", "sunday", 3),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2)
    ]
    
    print("Testing Edit Distance Solutions:")
    print("=" * 70)
    
    for i, (word1, word2, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: '{word1}' -> '{word2}'")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for long strings)
        if len(word1) + len(word2) <= 10:
            brute = edit_distance_bruteforce(word1, word2)
            print(f"Brute Force:      {brute:>3} {'✓' if brute == expected else '✗'}")
        
        memo = edit_distance_memoization(word1, word2)
        tab = edit_distance_tabulation(word1, word2)
        space_opt = edit_distance_space_optimized(word1, word2)
        one_row = edit_distance_one_row(word1, word2)
        
        print(f"Memoization:      {memo:>3} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab:>3} {'✓' if tab == expected else '✗'}")
        print(f"Space Optimized:  {space_opt:>3} {'✓' if space_opt == expected else '✗'}")
        print(f"One Row:          {one_row:>3} {'✓' if one_row == expected else '✗'}")
        
        # Show operations for small examples
        if len(word1) <= 5 and len(word2) <= 5:
            dist, operations = edit_distance_with_operations(word1, word2)
            print(f"Operations: {operations}")
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(3^max(m,n)), Space: O(max(m,n))")
    print("Memoization:      Time: O(m*n),        Space: O(m*n)")
    print("Tabulation:       Time: O(m*n),        Space: O(m*n)")
    print("Space Optimized:  Time: O(m*n),        Space: O(min(m,n))")
    print("One Row:          Time: O(m*n),        Space: O(min(m,n))")
    print("With Operations:  Time: O(m*n),        Space: O(m*n)")


if __name__ == "__main__":
    test_edit_distance()


"""
PATTERN RECOGNITION:
==================
This is a classic 2D string DP problem:
- Compare characters from two strings
- Three operations: insert, delete, replace
- dp[i][j] = min edit distance between word1[0:i] and word2[0:j]

KEY INSIGHTS:
============
1. If characters match, no operation needed: dp[i][j] = dp[i-1][j-1]
2. If different, try all three operations and take minimum
3. Base cases: empty string transformations require insertions/deletions
4. This is the classic Levenshtein distance algorithm

STATE DEFINITION:
================
dp[i][j] = minimum edit distance to transform word1[0:i] to word2[0:j]

RECURRENCE RELATION:
===================
If word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]
Else:
    dp[i][j] = 1 + min(
        dp[i-1][j],     # Delete from word1
        dp[i][j-1],     # Insert to word1
        dp[i-1][j-1]    # Replace in word1
    )

Base cases:
- dp[0][j] = j (insert j characters)
- dp[i][0] = i (delete i characters)

SPACE OPTIMIZATION:
==================
1. Two rows: O(min(m,n)) space
2. One row: O(min(m,n)) space with careful updates
3. Always make shorter string the column dimension

VARIANTS TO PRACTICE:
====================
- Longest Common Subsequence (1143) - related string DP
- Delete Operation for Two Strings (583) - only insert/delete
- Minimum ASCII Delete Sum (712) - weighted operations
- One Edit Distance (161) - check if exactly one edit away

INTERVIEW TIPS:
==============
1. Identify as classic edit distance problem
2. Draw small examples to understand recurrence
3. Explain three operations clearly
4. Show space optimization progression
5. Discuss operation reconstruction if needed
6. Handle edge cases (empty strings)
7. Mention applications (spell checkers, DNA sequence alignment)
"""
