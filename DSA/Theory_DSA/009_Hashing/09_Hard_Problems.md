# Hard Hashing Problems

## 1. Palindrome Pairs

**Description**: Find all pairs (i,j) where words[i] + words[j] is palindrome.

**Approach**: For each word, consider all splits; if reverse of one part exists and other part is palindrome, valid pair. Hash words to indices.

---

## 2. Substring with Concatenation of All Words

**Description**: Find all starting indices where substring is concatenation of every word exactly once.

**Approach**: Sliding window with word length; hash word counts; check each starting position.

---

## 3. Minimum Window Substring

**Description**: Smallest substring of s containing all chars of t.

**Approach**: Sliding window; hash to track char counts; expand/contract to satisfy.

---

## 4. Longest Substring with At Most K Distinct Characters

**Description**: Longest substring with at most k distinct chars.

**Approach**: Sliding window with hash for char counts; shrink when distinct > k.

---

## 5. Count Unique Characters of All Substrings

**Description**: Sum over all substrings of "unique char count" in each substring.

**Approach**: For each char, count substrings where it is unique (contribution method).

---

## 6. Subarrays with K Different Integers

**Description**: Count subarrays with exactly k distinct integers.

**Approach**: (At most K) - (At most K-1) using sliding window.

---

## 7. Minimum Window Subsequence

**Description**: Shortest substring of s that has t as subsequence.

**Approach**: DP or two pointers; hash for next occurrence can help.

---

## 8. Max Points on a Line

**Description**: Max collinear points.

**Approach**: For each point, hash slope (dx, dy) normalized; count max same slope.

---

## 9. First Missing Positive

**Description**: Find smallest missing positive integer in O(n) time O(1) space.

**Approach**: Index mapping; place each positive at its index; scan for first mismatch.

---

## 10. Trapping Rain Water II

**Description**: 3D version; water trapped in elevation map.

**Approach**: Min-heap from boundary; hash or visited set for processed cells.

---

## 11. Word Squares

**Description**: Arrange words so each row and column reads same word.

**Approach**: Backtrack; hash prefix to list of words; build square row by row.

---

## 12. Palindrome Pairs (Optimized)

**Description**: Same as above; optimize with Trie or rolling hash.

**Approach**: Trie of reversed words; for each word traverse and check remainder palindrome.

---

## 13. Count of Smaller Numbers After Self

**Description**: For each element, count smaller elements to the right.

**Approach**: Merge sort or Fenwick/BIT; hashing for coordinate compression.

---

## 14. Maximum XOR of Two Numbers in Array

**Description**: Find max XOR of any pair.

**Approach**: Trie with bits; for each number greedily choose opposite bit when possible.

---

## 15. Number of Matching Subsequences

**Description**: Count words that are subsequences of s.

**Approach**: Precompute next occurrence of each char; hash word to pointer in s.

---

## 16. Minimum Number of Refueling Stops

**Description**: Min stops to reach target with fuel stations.

**Approach**: Max-heap of fuel at passed stations; hash for station positions.

---

## 17. Longest Duplicate Substring

**Description**: Longest substring that appears at least twice.

**Approach**: Binary search on length; rolling hash (Rabin-Karp) for O(n) check.

---

## 18. Count Distinct Substrings

**Description**: Count distinct substrings of string.

**Approach**: Suffix array or Trie; hash set of substrings for simpler O(n^2) approach.

---

## 19. Repeated DNA Sequences

**Description**: Find 10-char sequences that appear more than once.

**Approach**: Rolling hash or encode as 2 bits per char; hash to count.

---

## 20. Group Shifted Strings

**Description**: Group strings that are shifts of each other (abc, bcd, cde).

**Approach**: Normalize by first char; use difference sequence as key.
