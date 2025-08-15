"""
LeetCode 139: Word Break
Difficulty: Medium
Category: Fibonacci & Linear DP

PROBLEM DESCRIPTION:
===================
Given a string s and a dictionary of strings wordDict, return true if s can be segmented 
into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

Example 1:
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Example 2:
Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.

Example 3:
Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false

Constraints:
- 1 <= s.length <= 300
- 1 <= wordDict.length <= 1000
- 1 <= wordDict[i].length <= 20
- s and wordDict[i] consist of only lowercase English letters.
- All the strings of wordDict are unique.
"""

def word_break_bruteforce(s, wordDict):
    """
    BRUTE FORCE APPROACH - RECURSION:
    ================================
    Try all possible ways to break the string using words from dictionary.
    
    Time Complexity: O(2^n) - exponential due to overlapping subproblems
    Space Complexity: O(n) - recursion stack depth
    """
    word_set = set(wordDict)  # Convert to set for O(1) lookup
    
    def can_break(start):
        if start == len(s):
            return True
        
        # Try all possible words starting from current position
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set and can_break(end):
                return True
        
        return False
    
    return can_break(0)


def word_break_memoization(s, wordDict):
    """
    DYNAMIC PROGRAMMING - TOP DOWN (MEMOIZATION):
    ============================================
    Use memoization to avoid recalculating same subproblems.
    
    Time Complexity: O(n^3) - n positions, n^2 substring operations
    Space Complexity: O(n) - memoization table + recursion stack
    """
    word_set = set(wordDict)
    memo = {}
    
    def can_break(start):
        if start == len(s):
            return True
        
        if start in memo:
            return memo[start]
        
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set and can_break(end):
                memo[start] = True
                return True
        
        memo[start] = False
        return False
    
    return can_break(0)


def word_break_tabulation(s, wordDict):
    """
    DYNAMIC PROGRAMMING - BOTTOM UP (TABULATION):
    =============================================
    Build solution from bottom up using 1D DP array.
    dp[i] = True if s[0:i] can be segmented using dictionary words
    
    Time Complexity: O(n^3) - nested loops + substring operation
    Space Complexity: O(n) - DP array
    """
    word_set = set(wordDict)
    n = len(s)
    
    # dp[i] = True if s[0:i] can be broken
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string can always be broken
    
    for i in range(1, n + 1):
        for j in range(i):
            # If s[0:j] can be broken and s[j:i] is in dictionary
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break  # Early termination
    
    return dp[n]


def word_break_optimized(s, wordDict):
    """
    OPTIMIZED DP WITH PRUNING:
    =========================
    Optimize by checking only valid word lengths and using early termination.
    
    Time Complexity: O(n * m * k) - n positions, m words, k average word length
    Space Complexity: O(n) - DP array
    """
    word_set = set(wordDict)
    n = len(s)
    
    # Get all possible word lengths for optimization
    word_lengths = set(len(word) for word in wordDict)
    max_word_length = max(word_lengths) if word_lengths else 0
    
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        # Only check substrings with valid word lengths
        for length in word_lengths:
            if length <= i:
                start = i - length
                if dp[start] and s[start:i] in word_set:
                    dp[i] = True
                    break
    
    return dp[n]


def word_break_bfs(s, wordDict):
    """
    BFS APPROACH:
    ============
    Use BFS to explore all possible segmentations level by level.
    
    Time Complexity: O(n^3) - similar to DP approaches
    Space Complexity: O(n) - queue and visited set
    """
    word_set = set(wordDict)
    queue = [0]  # Start positions to explore
    visited = set()
    
    while queue:
        start = queue.pop(0)
        
        if start == len(s):
            return True
        
        if start in visited:
            continue
        visited.add(start)
        
        # Try all possible words from current position
        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                queue.append(end)
    
    return False


def word_break_trie(s, wordDict):
    """
    TRIE-BASED APPROACH:
    ===================
    Build a trie from dictionary words for efficient prefix matching.
    
    Time Complexity: O(n^2 + m*k) - n^2 for DP, m*k for trie building
    Space Complexity: O(m*k + n) - trie storage + DP array
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_word = False
    
    # Build trie
    root = TrieNode()
    for word in wordDict:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
    
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    
    for i in range(1, n + 1):
        # Start from position i-1 and go backwards
        node = root
        for j in range(i - 1, -1, -1):
            char = s[j]
            if char not in node.children:
                break
            node = node.children[char]
            
            # If we found a complete word and prefix can be broken
            if node.is_word and dp[j]:
                dp[i] = True
                break
    
    return dp[n]


def word_break_with_segmentation(s, wordDict):
    """
    WORD BREAK WITH ACTUAL SEGMENTATION:
    ===================================
    Return not just whether segmentation is possible, but the actual segmentation.
    
    Time Complexity: O(n^3) - DP + segmentation reconstruction
    Space Complexity: O(n^2) - store all possible segmentations
    """
    word_set = set(wordDict)
    n = len(s)
    
    # dp[i] stores all possible segmentations for s[0:i]
    dp = [[] for _ in range(n + 1)]
    dp[0] = [[]]  # Empty segmentation for empty string
    
    for i in range(1, n + 1):
        for j in range(i):
            word = s[j:i]
            if word in word_set and dp[j]:
                # Add word to all existing segmentations at position j
                for segmentation in dp[j]:
                    dp[i].append(segmentation + [word])
    
    return len(dp[n]) > 0, dp[n]


# Test cases
def test_word_break():
    """Test all implementations with various inputs"""
    test_cases = [
        ("leetcode", ["leet", "code"], True),
        ("applepenapple", ["apple", "pen"], True),
        ("catsandog", ["cats", "dog", "sand", "and", "cat"], False),
        ("", ["a"], True),
        ("a", [], False),
        ("aaaaaaa", ["aaaa", "aaa"], True),
        ("cars", ["car", "ca", "rs"], True),
        ("goalspecial", ["go", "goal", "goals", "special"], True)
    ]
    
    print("Testing Word Break Solutions:")
    print("=" * 70)
    
    for i, (s, wordDict, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: s = '{s}', wordDict = {wordDict}")
        print(f"Expected: {expected}")
        
        # Test approaches (skip brute force for long strings)
        if len(s) <= 10:
            brute = word_break_bruteforce(s, wordDict.copy())
            print(f"Brute Force:      {brute} {'✓' if brute == expected else '✗'}")
        
        memo = word_break_memoization(s, wordDict.copy())
        tab = word_break_tabulation(s, wordDict.copy())
        opt = word_break_optimized(s, wordDict.copy())
        bfs = word_break_bfs(s, wordDict.copy())
        trie = word_break_trie(s, wordDict.copy())
        
        print(f"Memoization:      {memo} {'✓' if memo == expected else '✗'}")
        print(f"Tabulation:       {tab} {'✓' if tab == expected else '✗'}")
        print(f"Optimized:        {opt} {'✓' if opt == expected else '✗'}")
        print(f"BFS:              {bfs} {'✓' if bfs == expected else '✗'}")
        print(f"Trie:             {trie} {'✓' if trie == expected else '✗'}")
        
        # Show segmentations for positive cases
        if expected and len(s) <= 15:
            can_segment, segmentations = word_break_with_segmentation(s, wordDict.copy())
            if can_segment and segmentations:
                print(f"Segmentations: {segmentations[:3]}...")  # Show first 3
    
    print("\n" + "=" * 70)
    print("Complexity Analysis:")
    print("Brute Force:      Time: O(2^n),      Space: O(n)")
    print("Memoization:      Time: O(n^3),      Space: O(n)")
    print("Tabulation:       Time: O(n^3),      Space: O(n)")
    print("Optimized:        Time: O(n*m*k),    Space: O(n)")
    print("BFS:              Time: O(n^3),      Space: O(n)")
    print("Trie:             Time: O(n^2+m*k),  Space: O(m*k+n)")


if __name__ == "__main__":
    test_word_break()


"""
PATTERN RECOGNITION:
==================
This is a linear DP problem with string processing:
- For each position, check if string up to that position can be segmented
- dp[i] = True if s[0:i] can be broken using dictionary words
- Use previously computed results to build solution

KEY INSIGHTS:
============
1. dp[i] depends on all previous positions dp[j] where j < i
2. Need to check if s[j:i] forms a valid word from dictionary
3. Use set for O(1) word lookup instead of list
4. Can optimize by checking only valid word lengths

STATE DEFINITION:
================
dp[i] = True if string s[0:i] can be segmented using dictionary words

RECURRENCE RELATION:
===================
dp[i] = OR over all j < i of (dp[j] AND s[j:i] in wordDict)
Base case: dp[0] = True (empty string can always be segmented)

OPTIMIZATION TECHNIQUES:
=======================
1. Use set instead of list for dictionary
2. Early termination when dp[i] becomes True
3. Check only valid word lengths
4. Use trie for efficient prefix matching
5. BFS for exploring segmentations level by level

VARIANTS TO PRACTICE:
====================
- Word Break II (140) - return all possible segmentations
- Concatenated Words (472) - find words made of other words
- Word Squares (425) - form word squares from dictionary

INTERVIEW TIPS:
==============
1. Identify as string DP problem
2. Transform to: "can prefix be segmented?"
3. Use set for O(1) dictionary lookup
4. Show optimization with word length pruning
5. Discuss trie approach for advanced solution
6. Handle edge cases (empty string, empty dictionary)
7. Mention BFS as alternative approach
"""
