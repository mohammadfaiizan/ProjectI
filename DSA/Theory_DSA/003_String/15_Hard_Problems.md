# Hard String Problems

## 01. Regular Expression Matching
**Description**: Match string with pattern containing . and *.
**Approach**: DP: dp[i][j] = match s[:i] with p[:j], handle * (zero or more of preceding).

## 02. Wildcard Matching
**Description**: Match string with pattern containing ? and *.
**Approach**: DP similar to regex, * matches any sequence (greedy or DP).

## 03. Minimum Window Substring
**Description**: Smallest substring containing all chars of t.
**Approach**: Sliding window with frequency maps, expand then shrink.

## 04. Substring with Concatenation of All Words
**Description**: Find all starting indices of concatenation of words.
**Approach**: Sliding window over starts, check word-by-word with frequency.

## 05. Longest Valid Parentheses
**Description**: Length of longest valid parentheses substring.
**Approach**: Stack (store indices) or DP: dp[i] = longest valid ending at i.

## 06. Edit Distance (Levenshtein)
**Description**: Min insert/delete/replace to transform s1 to s2.
**Approach**: DP: dp[i][j] = min edits for s1[:i] to s2[:j].

## 07. Scramble String
**Description**: Can s1 be scrambled to form s2 (binary tree swap children)?
**Approach**: Recursion with memo: try all split points, check (a1,b1)+(a2,b2) or (a1,b2)+(a2,b1).

## 08. Distinct Subsequences
**Description**: Number of times t appears as subsequence of s.
**Approach**: DP: dp[i][j] = count for s[:i] and t[:j], match or skip.

## 09. Minimum Insertions to Make Palindrome
**Description**: Min chars to insert for palindrome.
**Approach**: n - longest palindromic subsequence.

## 10. Palindrome Partitioning II
**Description**: Min cuts so each part is palindrome.
**Approach**: DP: is_pal[i][j], then cuts[j] = min cuts for s[:j].

## 11. Word Ladder II
**Description**: All shortest transformation sequences from begin to end word.
**Approach**: BFS to find distance, DFS to reconstruct all paths.

## 12. Word Ladder
**Description**: Shortest transformation from begin to end (one char change).
**Approach**: BFS with queue, try all one-char variations.

## 13. Alien Dictionary
**Description**: Order of letters from sorted dictionary of alien language.
**Approach**: Build graph from adjacent word pairs, topological sort.

## 14. Longest Duplicate Substring
**Description**: Longest substring that appears at least twice.
**Approach**: Binary search on length + rolling hash (Rabin-Karp) to check.

## 15. Count Unique Characters of All Substrings
**Description**: Sum of unique char count over all substrings.
**Approach**: For each char, count substrings where it is unique (contribution = (i-prev)*(next-i)).

## 16. Palindrome Pairs
**Description**: Pairs (i,j) where words[i] + words[j] is palindrome.
**Approach**: For each word, check if reverse of prefix/suffix exists and remainder is palindrome.

## 17. Shortest Palindrome
**Description**: Prepend min chars to make palindrome.
**Approach**: Find longest palindromic prefix, prepend reverse of rest. Use KMP on s + "#" + reverse(s).
