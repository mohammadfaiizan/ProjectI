# Trie - Medium Problems

## 01. Word Search II

**Description**: Given 2D board and list of words, find all words that exist in board (adjacent cells, no reuse).

**Approach**: Build trie from words. Backtrack on grid: from each cell, traverse trie; when is_end, add word; prune when no child for current char.

---

## 02. Design Add and Search Words Data Structure

**Description**: Add words and search with '.' wildcard.

**Approach**: Trie + DFS for wildcard. When '.', try all children.

---

## 03. Prefix and Suffix Search

**Description**: Design structure to find word with given prefix and suffix. Return largest index.

**Approach**: For each word, store suffix + '#' + word for every suffix. Query: suffix + '#' + prefix. Store max index at each path.

---

## 04. Implement Magic Dictionary

**Description**: Build dict, search if any word matches with exactly one character different.

**Approach**: Trie. Search with one allowed mismatch: when chars match continue; when different, recurse with changed=True.

---

## 05. Replace Words

**Description**: Replace words in sentence with shortest root from dictionary.

**Approach**: Trie from roots. For each word, traverse until is_end, replace with path.

---

## 06. Map Sum Pairs

**Description**: Insert (key, val), sum(prefix) returns sum of values for keys with prefix.

**Approach**: Trie with value at node. Store delta on insert. Sum = sum of values in prefix subtree.

---

## 07. Word Squares

**Description**: Form word squares where k-th row and column read same string.

**Approach**: Trie stores words by prefix. Backtrack: build square row by row; for row i, prefix = column i of current rows; get words with that prefix from trie.

---

## 08. Palindrome Pairs

**Description**: Find all pairs (i, j) where words[i] + words[j] is palindrome.

**Approach**: Trie of reversed words. For each word, traverse trie; when at is_end, check if remainder of word is palindrome. Also check words in subtree when current word exhausted.

---

## 09. Maximum XOR of Two Numbers in an Array

**Description**: Find maximum XOR of any two numbers in array.

**Approach**: Bitwise trie. Insert numbers as 32-bit binary. For each number, traverse trie choosing opposite bit when available to maximize XOR.

---

## 10. Maximum XOR With an Element From Array

**Description**: For each query (x, m), find max XOR of x with any array element <= m.

**Approach**: Sort queries by m. Process in order, inserting numbers into trie as m increases. Query max XOR for x.

---

## 11. Search Suggestions System

**Description**: As user types, suggest top 3 products (by prefix match).

**Approach**: Sort products. Binary search for prefix. Or trie with DFS to collect, sort by relevance.

---

## 12. Stream of Characters

**Description**: Check if any suffix of stream is in words.

**Approach**: Trie of reversed words. Maintain stream. For each new char, check reversed suffixes.

---

## 13. Word Break

**Description**: Determine if string can be segmented into dictionary words.

**Approach**: Trie + DP. Trie for fast prefix lookup. dp[i] = can segment s[:i]. For each i where dp[i], try all prefixes starting at i via trie.

---

## 14. Concatenated Words

**Description**: Find all words that can be formed by concatenating two or more shorter words from same list.

**Approach**: Trie of all words. For each word, DFS: at each position check if prefix is word and remainder can be formed.

---

## 15. Add and Search Word - Data Structure Design

**Description**: Same as Design Add and Search Words. Add word, search with '.' wildcard.

**Approach**: Trie with DFS for wildcard search.

---

## 16. Shortest Word Distance II

**Description**: Design class with list of words. Query shortest distance between two words.

**Approach**: HashMap word to sorted list of indices. Two pointers or binary search for min |i - j|.

---

## 17. Shortest Word Distance III

**Description**: Shortest distance between word1 and word2 when they can be same.

**Approach**: Track last indices of both. When same word, use prev and current.

---

## 18. CamelCase Matching

**Description**: For each query, check if it matches pattern (uppercase must match, lowercase can match multiple).

**Approach**: For each query, two pointers: pattern pointer advances when match; if query has uppercase not matching, return false.

---

## 19. Count Pairs With Given XOR

**Description**: Count pairs (i, j) with nums[i] XOR nums[j] == target.

**Approach**: For each x, need y = x XOR target. Use HashMap to count. Trie for range XOR variants.

---

## 20. Substring With Concatenation of All Words

**Description**: Find all starting indices where substring is concatenation of all words (each exactly once).

**Approach**: Sliding window with word length. HashMap word counts. Trie can store words for matching.

---

## 21. Word Ladder

**Description**: Shortest transformation from begin to end word, changing one letter at a time, each step must be in word list.

**Approach**: BFS. Trie can optimize neighbor finding: for each position, try all 26 letters, check if in trie.

---

## 22. Word Ladder II

**Description**: Find all shortest transformation sequences from begin to end.

**Approach**: BFS to find distance, then DFS to reconstruct paths. Trie for neighbor lookup.

---

## 23. Group Anagrams

**Description**: Group words that are anagrams.

**Approach**: Sort each word as key, group. Trie can store by sorted form.

---

## 24. Longest Word in Dictionary Through Deleting

**Description**: Find longest word in dictionary that is subsequence of string s.

**Approach**: For each dict word, check if subsequence of s. Sort by length desc and lexicographic. Trie: build from s for subsequence matching.

---

## 25. Number of Matching Subsequences

**Description**: Count how many words in words are subsequences of string s.

**Approach**: Preprocess s: for each char, list indices. For each word, binary search next occurrence. Trie of subsequences possible.

---

## Hard Problems

## H01. Word Search II

**Description**: Find all words from dictionary in 2D grid. Same as medium but often classified hard.

**Approach**: Trie + backtracking. Build trie, prune during backtrack when no words with current prefix.

---

## H02. Palindrome Pairs

**Description**: Find all (i, j) where words[i] + words[j] is palindrome.

**Approach**: Trie of reversed words. For each word, traverse trie; at each is_end check remainder palindrome; when word exhausted, collect all words in subtree with palindrome suffix.

---

## H03. Maximum XOR of Two Numbers in an Array

**Description**: Max XOR of any two numbers.

**Approach**: Binary trie. Insert bits MSB first. For each number, greedily choose opposite bit.

---

## H04. Word Squares

**Description**: Form NxN grid where each row and column is a valid word.

**Approach**: Trie by prefix. Backtrack row by row. Prefix for row i = column i of current grid. Get candidates from trie.

---

## H05. Prefix and Suffix Search

**Description**: Find word with prefix and suffix, return max index.

**Approach**: Store suffix + '#' + word for each suffix of each word. Query suffix + '#' + prefix.

---

## H06. Count Pairs With XOR in a Range

**Description**: Count pairs (i, j) where low <= nums[i] XOR nums[j] <= high.

**Approach**: Count pairs with XOR < high+1 minus count with XOR < low. Use binary trie with count at nodes.

---

## H07. Maximum XOR With an Element From Array

**Description**: Queries (x, m): max XOR of x with array element <= m.

**Approach**: Sort queries by m. Incrementally build trie. For each query, find max XOR in trie.

---

## H08. Word Search II (Optimized)

**Description**: Same as medium but with large dictionary. Need pruning.

**Approach**: Trie with remove during backtrack to avoid duplicate finds. Or mark node as visited.

---

## H09. Count Distinct Substrings

**Description**: Count distinct substrings of string.

**Approach**: Suffix trie. Each path from root is unique substring. Count nodes (excluding root) or use suffix array.

---

## H10. Longest Duplicate Substring

**Description**: Find longest substring that appears at least twice.

**Approach**: Binary search on length + rolling hash or suffix array. Trie of suffixes for each length.

---

## H11. Multi-Search

**Description**: Given string b and array of small strings T, find all occurrences of each T[i] in b.

**Approach**: Build trie from T. For each starting position in b, traverse trie and record matches.

---

## H12. Word Rectangle

**Description**: Find largest rectangle of letters such that each row and column is a word.

**Approach**: Trie for rows and columns. Try dimensions, backtrack with trie validation.

---

## H13. Maximum XOR of Two Numbers in a Tree

**Description**: Tree with values on nodes. Find max XOR of any two node values.

**Approach**: DFS from root. At each node, insert path XOR into trie. Query max XOR with current path XOR.

---

## H14. Count Substrings With One Distinct Letter

**Description**: Count substrings with exactly one distinct character.

**Approach**: Group consecutive same chars. Each group of length n contributes n*(n+1)/2. Trie not typical.

---

## H15. Substring With Largest Variance

**Description**: Find substring with largest difference between max and min frequency of two chars.

**Approach**: Kadane variant. For each pair of chars (a, b), treat as +1 and -1, find max subarray sum. Trie not typical.
