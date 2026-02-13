# Easy Bit Manipulation Problems

## 1. Number of 1 Bits
**Description:** Count the number of set bits in an unsigned integer.
**Approach:** Use Brian Kernighan's algorithm (n &= n-1) or loop with n & 1.

## 2. Power of Two
**Description:** Determine if an integer is a power of 2.
**Approach:** n > 0 and (n & (n-1)) == 0.

## 3. Power of Four
**Description:** Determine if an integer is a power of 4.
**Approach:** Power of two check plus (n & 0xAAAAAAAA) == 0 to ensure odd bit position.

## 4. Binary Number with Alternating Bits
**Description:** Check if binary representation has alternating 0s and 1s.
**Approach:** XOR with right-shifted self; result should be of form 2^k - 1.

## 5. Hamming Distance
**Description:** Number of positions where bits differ between two integers.
**Approach:** XOR both numbers and count set bits in result.

## 6. Complement of Base 10 Integer
**Description:** Return complement (flip all bits) of a positive integer.
**Approach:** XOR with mask of (2^bit_length - 1).

## 7. Single Number
**Description:** Every element appears twice except one. Find the unique element.
**Approach:** XOR all elements; duplicates cancel out.

## 8. Missing Number
**Description:** Array of n distinct numbers in [0, n]. Find the missing one.
**Approach:** XOR 0 to n with all array elements.

## 9. Decode XORed Array
**Description:** Reconstruct original array from encoded where encoded[i] = arr[i] XOR arr[i+1].
**Approach:** arr[0] given; arr[i+1] = arr[i] XOR encoded[i].

## 10. XOR Operation in an Array
**Description:** Build array where arr[i] = start + 2*i, return XOR of all.
**Approach:** Direct XOR of all elements or use XOR range pattern.

## 11. Count the Number of Consistent Strings
**Description:** Count strings that contain only allowed characters.
**Approach:** Use bitmask for allowed chars; check each string's mask is subset.

## 12. Reverse Bits
**Description:** Reverse bits of 32-bit unsigned integer.
**Approach:** Extract bits one by one and build reversed number.

## 13. Number of Steps to Reduce a Number to Zero
**Description:** Steps: if even divide by 2, else subtract 1.
**Approach:** Count set bits (subtractions) + bit length (divisions).

## 14. Subtract the Product and Sum of Digits
**Description:** Product of digits minus sum of digits.
**Approach:** Not bit-specific but digit manipulation; can use bit tricks for digit extraction.

## 15. Check if Number is a Sum of Powers of Three
**Description:** Can n be expressed as sum of distinct powers of 3?
**Approach:** Convert to base 3; digits must be 0 or 1 (no 2).

## 16. Binary Gap
**Description:** Longest distance between two consecutive 1s in binary.
**Approach:** Track positions of 1s, compute gaps.

## 17. Convert to Base -2
**Description:** Represent integer in base -2.
**Approach:** Similar to base conversion; remainder can be negative, adjust.

## 18. Prime Number of Set Bits in Binary Representation
**Description:** Count numbers in range [L,R] with prime number of set bits.
**Approach:** For each number count bits, check if count is prime.

## 19. Number Complement
**Description:** Flip all bits of positive integer (leading zeros not flipped).
**Approach:** XOR with (1 << bit_length) - 1.

## 20. Find the Difference
**Description:** String t is s with one random char added. Find the added char.
**Approach:** XOR all character codes; duplicates cancel.

## 21. Counting Bits
**Description:** Return array where ans[i] = number of 1s in binary of i.
**Approach:** DP: count[i] = count[i >> 1] + (i & 1).

## 22. Sum of Two Integers
**Description:** Calculate a + b without using + or -.
**Approach:** XOR for sum, AND << 1 for carry; repeat until carry is 0.

## 23. Maximum Product of Word Lengths
**Description:** Max product of lengths of two words with no common letters.
**Approach:** Bitmask each word (a=bit0, b=bit1...); max len1*len2 where mask1 & mask2 == 0.

## 24. Check if String Contains All Binary Codes of Size K
**Description:** Does string contain all 2^k binary codes of length k as substrings?
**Approach:** Rolling hash or bitmask; collect all k-length substrings as integers.

## 25. Minimum Flips to Make a OR b Equal to c
**Description:** Flip minimum bits in a or b so that (a OR b) == c.
**Approach:** For each bit: if c has 0, both a and b must be 0; if c has 1, at least one of a,b must be 1.

## 26. Sort Integers by The Number of 1 Bits
**Description:** Sort array by popcount, then by value.
**Approach:** Use (popcount(x), x) as sort key.

## 27. Minimum Bit Flips to Convert Number
**Description:** Minimum bits to flip to convert start to goal.
**Approach:** XOR start and goal; count set bits.

## 28. XOR Queries of a Subarray
**Description:** Answer queries: XOR of arr[l..r].
**Approach:** Prefix XOR array; query = prefix[r+1] ^ prefix[l].
