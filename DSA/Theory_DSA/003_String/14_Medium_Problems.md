# Medium String Problems

## 01. Longest Substring Without Repeating Characters
**Description**: Find length of longest substring with all unique chars.
**Approach**: Sliding window with hash map of char to last index.

## 02. Longest Palindromic Substring
**Description**: Find longest palindrome substring.
**Approach**: Expand around center for each index (odd and even length), or Manacher O(n).

## 03. Zigzag Conversion
**Description**: Write string in zigzag pattern, read row by row.
**Approach**: Simulate row indices: 0,1,...,numRows-1,numRows-2,...,0.

## 04. String to Integer (atoi)
**Description**: Parse integer from string with overflow.
**Approach**: Strip, sign, accumulate digits, clamp to 32-bit.

## 05. Letter Combinations of Phone Number
**Description**: All letter combos for digit string (2=abc, 3=def, etc).
**Approach**: Backtracking or iterative BFS.

## 06. Generate Parentheses
**Description**: Generate all valid n pairs of parentheses.
**Approach**: Backtrack: add "(" if open < n, add ")" if close < open.

## 07. Group Anagrams
**Description**: Group strings that are anagrams.
**Approach**: Use sorted string or frequency tuple as key in defaultdict.

## 08. Decode Ways
**Description**: Number of ways to decode digit string (1=A,...,26=Z).
**Approach**: DP: dp[i] = ways for s[:i], consider 1-digit and 2-digit.

## 09. Word Break
**Description**: Can string be segmented into dictionary words?
**Approach**: DP: dp[i] = True if s[:i] can be broken.

## 10. Longest Repeating Character Replacement
**Description**: Longest substring with same char after at most k replacements.
**Approach**: Sliding window, track max freq in window, shrink when (len - max_freq) > k.

## 11. Find All Anagrams in String
**Description**: Starting indices where anagram of p exists in s.
**Approach**: Sliding window of len(p), compare frequency with p.

## 12. Permutation in String
**Description**: Does s2 contain permutation of s1?
**Approach**: Sliding window, check if window freq equals s1 freq.

## 13. Minimum Window Substring
**Description**: Smallest substring of s containing all chars of t.
**Approach**: Sliding window, expand until valid, shrink from left, track min.

## 14. Substring with Concatenation of All Words
**Description**: Find indices where concatenation of all words appears.
**Approach**: Sliding window over possible starts, check each word in window.

## 15. Longest Palindromic Subsequence
**Description**: Length of longest palindromic subsequence.
**Approach**: DP: dp[i][j] = LPS of s[i:j+1], recurse on ends.

## 16. Palindromic Substrings
**Description**: Count all palindromic substrings.
**Approach**: Expand around center for each index (odd and even).

## 17. Encode and Decode Strings
**Description**: Serialize list of strings for transmission.
**Approach**: Length-prefixed: "4#word" format, parse by reading length then chars.

## 18. Reorganize String
**Description**: Reorder so no two adjacent same.
**Approach**: Max-heap by frequency, alternate most frequent with others.

## 19. Compare Version Numbers
**Description**: Compare two version strings (1.0.1 vs 1.0.0).
**Approach**: Split by ".", pad with zeros, compare numerically.

## 20. Multiply Strings
**Description**: Multiply two numbers as strings.
**Approach**: Digit-by-digit multiplication, store in array, handle carries.

## 21. Simplify Path
**Description**: Simplify Unix path (remove . and .., collapse slashes).
**Approach**: Split by "/", use stack: push for dir, pop for "..", ignore ".".

## 22. Basic Calculator II
**Description**: Evaluate expression with +, -, *, /.
**Approach**: Parse and evaluate, handle * and / first (two-pass or stack).

## 23. Restore IP Addresses
**Description**: All valid IP addresses from string.
**Approach**: Backtrack: place 3 dots, check each segment 0-255 and no leading zeros.

## 24. Word Search
**Description**: Does grid contain word (adjacent cells)?
**Approach**: DFS from each cell, backtrack with visited set.

## 25. Implement Trie
**Description**: Implement prefix tree (insert, search, startsWith).
**Approach**: Node with children dict, leaf marker for complete words.
