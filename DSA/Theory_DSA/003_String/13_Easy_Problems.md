# Easy String Problems

## 01. Valid Palindrome
**Description**: Check if string reads same forward and backward, ignoring non-alphanumeric and case.
**Approach**: Two pointers from both ends, skip non-alphanumeric, compare lowercase.

## 02. Valid Anagram
**Description**: Determine if two strings are anagrams.
**Approach**: Sort both and compare, or use frequency count (Counter/dict).

## 03. First Unique Character
**Description**: Find index of first non-repeating character.
**Approach**: Two passes: count frequency, then find first with count 1.

## 04. Reverse String
**Description**: Reverse string in-place.
**Approach**: Two pointers swap from both ends.

## 05. Reverse Vowels
**Description**: Reverse only vowels in string.
**Approach**: Two pointers, swap when both point to vowels.

## 06. String to Integer (atoi)
**Description**: Convert string to integer with overflow handling.
**Approach**: Strip, handle sign, iterate digits, clamp to 32-bit range.

## 07. Implement strStr
**Description**: Find first occurrence of needle in haystack.
**Approach**: Brute force O(n*m) or KMP O(n+m).

## 08. Longest Common Prefix
**Description**: Find longest common prefix of array of strings.
**Approach**: Vertical scan (char by char) or horizontal (compare with first string).

## 09. Valid Parentheses
**Description**: Check if brackets are balanced.
**Approach**: Stack, push open, pop and match on close.

## 10. Roman to Integer
**Description**: Convert Roman numeral to integer.
**Approach**: Scan left to right, subtract if next is larger (e.g., IV = 4).

## 11. Integer to Roman
**Description**: Convert integer to Roman numeral.
**Approach**: Greedy with value-symbol pairs in descending order.

## 12. Count and Say
**Description**: Generate nth term of count-and-say sequence.
**Approach**: Iterate n-1 times, each time replace runs with count+digit.

## 13. Length of Last Word
**Description**: Return length of last word in string.
**Approach**: Strip, split, return len of last element. Or traverse backward.

## 14. Add Binary
**Description**: Add two binary strings.
**Approach**: Simulate addition with carry from right to left.

## 15. Excel Sheet Column Title
**Description**: Convert number to Excel column (1=A, 27=AA).
**Approach**: Repeated modulo 26, prepend char. Note: 1-indexed so subtract 1 before mod.

## 16. Excel Sheet Column Number
**Description**: Convert Excel column to number.
**Approach**: Accumulate base-26: result = result*26 + (ord(c)-ord('A')+1).

## 17. Isomorphic Strings
**Description**: Check if two strings have same character mapping.
**Approach**: Two dicts: s->t and t->s, ensure bijection.

## 18. Word Pattern
**Description**: Check if pattern matches word sequence (bijection).
**Approach**: Same as isomorphic: pattern char <-> word mapping.

## 19. Reverse Words in String
**Description**: Reverse order of words, trim extra spaces.
**Approach**: Split, reverse, join. Or two-pointer reverse words then whole string.

## 20. Ransom Note
**Description**: Check if ransom can be formed from magazine chars.
**Approach**: Count magazine chars, decrement for ransom, ensure all non-negative.

## 21. Valid Palindrome II
**Description**: Can string be palindrome with at most one deletion?
**Approach**: Two pointers, on mismatch try skip left or skip right.

## 22. Detect Capital
**Description**: Check if word uses capitals correctly (all, first only, none).
**Approach**: Count uppercase, check cases: all caps, first only, or none.

## 23. Student Attendance Record I
**Description**: Reward if no more than one A and no three consecutive L.
**Approach**: Count A, check for "LLL" substring.

## 24. Judge Route Circle
**Description**: Robot returns to origin after moves.
**Approach**: Net U-D and L-R must be zero.

## 25. Repeated Substring Pattern
**Description**: Can string be formed by repeating a substring?
**Approach**: Concatenate s+s, remove first and last char, check if s in result.
