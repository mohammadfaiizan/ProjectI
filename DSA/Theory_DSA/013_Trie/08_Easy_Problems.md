# Trie - Easy Problems

## 01. Implement Trie (Prefix Tree)

**Description**: Implement a trie with insert, search, and startsWith methods.

**Approach**: Standard trie with HashMap or array children. Insert character by character, set is_end at last node. Search returns true only if path exists and is_end. startsWith returns true if path exists.

---

## 02. Design Add and Search Words Data Structure

**Description**: Design a structure that supports adding words and searching with '.' as wildcard matching any single character.

**Approach**: Trie structure. For search with '.', use DFS: when char is '.', recurse on all children; otherwise follow the specific child.

---

## 03. Longest Common Prefix

**Description**: Find the longest common prefix string amongst an array of strings.

**Approach**: Build trie from all words. Traverse from root while node has exactly one child and is not a word end. The path gives LCP.

---

## 04. Replace Words

**Description**: In a sentence, replace each word with its shortest root from dictionary if it has one.

**Approach**: Build trie from dictionary roots. For each word in sentence, traverse trie; when is_end is hit, that prefix is the replacement.

---

## 05. Map Sum Pairs

**Description**: Design a map that supports insert(key, val) and sum(prefix) returning sum of all values for keys with that prefix.

**Approach**: Trie with value/count at each node. On insert, compute delta from old value. Sum = sum of values in subtree under prefix.

---

## 06. Index Pairs of a String

**Description**: Given text and list of words, find all [start, end] pairs where text[start:end] is in words.

**Approach**: Build trie from words. For each starting index i, traverse trie with text[i:], record (i, j) when is_end at position j.

---

## 07. Count Prefixes of a Given String

**Description**: Count how many words in array are prefix of given string.

**Approach**: Build trie from words. Traverse string in trie; at each step count words (is_end) under current node. Or check each word with string.startswith(word).

---

## 08. Stream of Characters

**Description**: Design StreamChecker that receives stream of chars and checks if any suffix of stream so far is in words.

**Approach**: Build trie from reversed words. Maintain recent chars. For each new char, check reversed suffixes against trie.

---

## 09. Search Suggestions System

**Description**: As user types, suggest top 3 products with matching prefix.

**Approach**: Sort products, use binary search for prefix. Or build trie, DFS to collect words, sort by frequency and take top 3.

---

## 10. Shortest Word Distance

**Description**: Given list of words and two words, find shortest distance between their occurrences.

**Approach**: Store indices in list per word. Two pointers to find min difference. Trie can store word positions.

---

## 11. Word Pattern

**Description**: Check if pattern matches string (bijection between pattern chars and words).

**Approach**: Two HashMaps for mapping. Trie not typical.

---

## 12. Isomorphic Strings

**Description**: Check if two strings are isomorphic (character mapping).

**Approach**: Build mapping both ways. Trie not needed.

---

## 13. Valid Anagram

**Description**: Check if two strings are anagrams.

**Approach**: Count array or sort. Trie can store sorted anagram groups.

---

## 14. First Unique Character in a String

**Description**: Find index of first non-repeating character.

**Approach**: Count frequency, scan for first with count 1. Trie can track first occurrence.

---

## 15. Find the Difference

**Description**: String t is s with one extra letter. Find the extra letter.

**Approach**: XOR or count array. Trie not needed.

---

## 16. Reverse Words in a String III

**Description**: Reverse each word in string, keep word order.

**Approach**: Split, reverse each, join. Trie not needed.

---

## 17. Count Binary Substrings

**Description**: Count contiguous substrings with same number of 0s and 1s.

**Approach**: Group consecutive same chars, adjacent groups contribute min(count1, count2). Trie not needed.

---

## 18. To Lower Case

**Description**: Convert string to lowercase.

**Approach**: Built-in or char-by-char. Trie not needed.

---

## 19. Robot Return to Origin

**Description**: Check if moves return robot to origin.

**Approach**: Count U-D and L-R. Trie not needed.

---

## 20. Defanging an IP Address

**Description**: Replace '.' with '[.]' in IP string.

**Approach**: String replace. Trie not needed.

---

## 21. Jewels and Stones

**Description**: Count how many chars of stones are in jewels.

**Approach**: Set of jewels, count stones in set. Trie can store jewels for prefix matching.

---

## 22. Unique Morse Code Words

**Description**: Count unique morse code representations of words.

**Approach**: Convert each word to morse, add to set, return size. Trie can store morse strings.

---

## 23. Goat Latin

**Description**: Apply goat latin rules to sentence.

**Approach**: Split, transform each word by rules, join. Trie not needed.

---

## 24. Buddy Strings

**Description**: Check if swap of two chars in A can make B.

**Approach**: If A == B, need duplicate char. Else exactly two positions differ and A[i]=B[j], A[j]=B[i]. Trie not needed.

---

## 25. Longest Uncommon Subsequence I

**Description**: Find longest uncommon subsequence of two strings.

**Approach**: If A == B return -1. Else return max(len(A), len(B)). Trie not needed.
