# Easy Hashing Problems

## 1. Two Sum

**Description**: Given an array and target, return indices of two numbers that add up to target.

**Approach**: Hash map storing value to index. For each element, check if complement (target - value) exists.

---

## 2. Contains Duplicate

**Description**: Return true if array has any duplicate element.

**Approach**: Compare len(nums) with len(set(nums)) or use set to track seen elements.

---

## 3. Valid Anagram

**Description**: Check if two strings are anagrams.

**Approach**: Count characters in both strings; compare counts or use sorted strings as key.

---

## 4. First Unique Character in a String

**Description**: Find index of first non-repeating character.

**Approach**: Count frequency with Counter, scan for first char with count 1.

---

## 5. Intersection of Two Arrays

**Description**: Return unique elements common to both arrays.

**Approach**: Convert both to sets, return intersection.

---

## 6. Intersection of Two Arrays II

**Description**: Return intersection with duplicates preserved (min count).

**Approach**: Count one array, iterate second and decrement count when found.

---

## 7. Happy Number

**Description**: Determine if number reaches 1 when repeatedly summing squares of digits.

**Approach**: Set to detect cycle; if we see a number again, not happy.

---

## 8. Isomorphic Strings

**Description**: Check if two strings have one-to-one character mapping.

**Approach**: Two hash maps: s->t and t->s. Verify consistency for each pair.

---

## 9. Word Pattern

**Description**: Check if pattern matches space-separated words bijectively.

**Approach**: Same as isomorphic: pattern char to word and word to pattern char.

---

## 10. Contains Duplicate II

**Description**: Check if duplicate exists within distance k.

**Approach**: Sliding window with set or dict storing last index per value.

---

## 11. Ransom Note

**Description**: Check if magazine has enough letters to form ransom note.

**Approach**: Count magazine chars, decrement for each ransom char.

---

## 12. Jewels and Stones

**Description**: Count how many stones are jewels.

**Approach**: Set of jewels, count stones in set.

---

## 13. Find the Difference

**Description**: Find the one extra character in string t compared to s.

**Approach**: Count chars in both, find the one with different count.

---

## 14. Single Number

**Description**: Find the single non-duplicate in array where others appear twice.

**Approach**: XOR all elements (no hash needed) or use Counter.

---

## 15. Majority Element

**Description**: Find element appearing more than n/2 times.

**Approach**: Boyer-Moore voting or Counter.most_common(1).

---

## 16. Find All Numbers Disappeared in an Array

**Description**: Array 1..n, some missing; return missing numbers.

**Approach**: Mark indices by negating; unmarked indices are missing.

---

## 17. Find All Duplicates in an Array

**Description**: Array 1..n, some appear twice; return duplicates.

**Approach**: Same marking; when we try to mark already negative, it is duplicate.

---

## 18. Keyboard Row

**Description**: Return words that can be typed using one keyboard row.

**Approach**: Map each letter to row number; check all chars in word same row.

---

## 19. Distribute Candies

**Description**: Max distinct candy types sister can get (n/2 max).

**Approach**: min(len(set(candyType)), n//2).

---

## 20. Set Mismatch

**Description**: Array 1..n with one duplicate and one missing; return [duplicate, missing].

**Approach**: Find duplicate via marking; missing = expected sum - actual sum + duplicate.

---

## 21. Number of Good Pairs

**Description**: Count pairs (i,j) with i<j and nums[i]==nums[j].

**Approach**: Count frequencies; each count c contributes c*(c-1)//2.

---

## 22. Count Pairs with Given Difference K

**Description**: Count pairs with absolute difference k.

**Approach**: Counter; for each x, add count of x+k and x-k (handle k=0 separately).

---

## 23. Find Common Characters

**Description**: Common characters across all strings with multiplicity.

**Approach**: Intersect Counter of each string (use & operator).

---

## 24. Subdomain Visit Count

**Description**: Parse "cnt domain" and aggregate by domain and subdomains.

**Approach**: Split domain by dots, add count to each suffix subdomain.

---

## 25. Most Common Word

**Description**: Most frequent word not in banned list.

**Approach**: Regex to extract words, Counter, filter banned.

---

## 26. Unique Morse Code Words

**Description**: Count unique morse representations of words.

**Approach**: Map each word to morse string, add to set, return len(set).

---

## 27. Buddy Strings

**Description**: Can we swap two chars in A to get B?

**Approach**: If A==B, need at least one duplicate. Else need exactly two mismatches that swap correctly.

---

## 28. Uncommon Words from Two Sentences

**Description**: Words that appear exactly once across both sentences.

**Approach**: Count all words; return those with count 1.
