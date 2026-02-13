# Easy Bit Manipulation Problems

## 1. Number of 1 Bits

**Description:** Count the number of set bits in an unsigned integer.

**Approach:** Use Brian Kernighan's algorithm (n &= n-1) or loop with n & 1.

```python
def hammingWeight(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count
```

Time: O(k) k=set bits | Space: O(1)

---

## 2. Power of Two

**Description:** Determine if an integer is a power of 2.

**Approach:** n > 0 and (n & (n-1)) == 0.

```python
def isPowerOfTwo(n):
    return n > 0 and (n & (n - 1)) == 0
```

Time: O(1) | Space: O(1)

---

## 3. Power of Four

**Description:** Determine if an integer is a power of 4.

**Approach:** Power of two check plus (n & 0xAAAAAAAA) == 0 to ensure odd bit position.

```python
def isPowerOfFour(n):
    return n > 0 and (n & (n - 1)) == 0 and (n & 0xAAAAAAAA) == 0
```

Time: O(1) | Space: O(1)

---

## 4. Binary Number with Alternating Bits

**Description:** Check if binary representation has alternating 0s and 1s.

**Approach:** XOR with right-shifted self; result should be of form 2^k - 1.

```python
def hasAlternatingBits(n):
    x = n ^ (n >> 1)
    return (x & (x + 1)) == 0
```

Time: O(1) | Space: O(1)

---

## 5. Hamming Distance

**Description:** Number of positions where bits differ between two integers.

**Approach:** XOR both numbers and count set bits in result.

```python
def hammingDistance(x, y):
    return bin(x ^ y).count('1')
```

Time: O(1) | Space: O(1)

---

## 6. Complement of Base 10 Integer

**Description:** Return complement (flip all bits) of a positive integer.

**Approach:** XOR with mask of (2^bit_length - 1).

```python
def bitwiseComplement(n):
    if n == 0:
        return 1
    mask = (1 << n.bit_length()) - 1
    return n ^ mask
```

Time: O(1) | Space: O(1)

---

## 7. Single Number

**Description:** Every element appears twice except one. Find the unique element.

**Approach:** XOR all elements; duplicates cancel out.

```python
def singleNumber(nums):
    res = 0
    for x in nums:
        res ^= x
    return res
```

Time: O(n) | Space: O(1)

---

## 8. Missing Number

**Description:** Array of n distinct numbers in [0, n]. Find the missing one.

**Approach:** XOR 0 to n with all array elements.

```python
def missingNumber(nums):
    res = len(nums)
    for i, x in enumerate(nums):
        res ^= i ^ x
    return res
```

Time: O(n) | Space: O(1)

---

## 9. Decode XORed Array

**Description:** Reconstruct original array from encoded where encoded[i] = arr[i] XOR arr[i+1].

**Approach:** arr[0] given; arr[i+1] = arr[i] XOR encoded[i].

```python
def decode(encoded, first):
    res = [first]
    for x in encoded:
        res.append(res[-1] ^ x)
    return res
```

Time: O(n) | Space: O(1)

---

## 10. XOR Operation in an Array

**Description:** Build array where arr[i] = start + 2*i, return XOR of all.

**Approach:** Direct XOR of all elements or use XOR range pattern.

```python
def xorOperation(n, start):
    res = 0
    for i in range(n):
        res ^= start + 2 * i
    return res
```

Time: O(n) | Space: O(1)

---

## 11. Count the Number of Consistent Strings

**Description:** Count strings that contain only allowed characters.

**Approach:** Use bitmask for allowed chars; check each string's mask is subset.

```python
def countConsistentStrings(allowed, words):
    mask = 0
    for c in allowed:
        mask |= 1 << (ord(c) - 97)
    return sum(1 for w in words if all((mask >> (ord(c) - 97)) & 1 for c in w))
```

Time: O(n * m) | Space: O(1)

---

## 12. Reverse Bits

**Description:** Reverse bits of 32-bit unsigned integer.

**Approach:** Extract bits one by one and build reversed number.

```python
def reverseBits(n):
    res = 0
    for _ in range(32):
        res = (res << 1) | (n & 1)
        n >>= 1
    return res
```

Time: O(32) | Space: O(1)

---

## 13. Number of Steps to Reduce a Number to Zero

**Description:** Steps: if even divide by 2, else subtract 1.

**Approach:** Count set bits (subtractions) + bit length (divisions).

```python
def numberOfSteps(num):
    steps = 0
    while num:
        steps += 1 + (num & 1)
        num >>= 1
    return max(0, steps - 1)
```

Time: O(log n) | Space: O(1)

---

## 14. Subtract the Product and Sum of Digits

**Description:** Product of digits minus sum of digits.

**Approach:** Not bit-specific but digit manipulation; can use bit tricks for digit extraction.

```python
def subtractProductAndSum(n):
    prod, s = 1, 0
    while n:
        n, d = divmod(n, 10)
        prod *= d
        s += d
    return prod - s
```

Time: O(log n) | Space: O(1)

---

## 15. Check if Number is a Sum of Powers of Three

**Description:** Can n be expressed as sum of distinct powers of 3?

**Approach:** Convert to base 3; digits must be 0 or 1 (no 2).

```python
def checkPowersOfThree(n):
    while n:
        if n % 3 == 2:
            return False
        n //= 3
    return True
```

Time: O(log n) | Space: O(1)

---

## 16. Binary Gap

**Description:** Longest distance between two consecutive 1s in binary.

**Approach:** Track positions of 1s, compute gaps.

```python
def binaryGap(n):
    last, res = None, 0
    for i in range(32):
        if (n >> i) & 1:
            if last is not None:
                res = max(res, i - last)
            last = i
    return res
```

Time: O(32) | Space: O(1)

---

## 17. Convert to Base -2

**Description:** Represent integer in base -2.

**Approach:** Similar to base conversion; remainder can be negative, adjust.

```python
def baseNeg2(n):
    if n == 0:
        return "0"
    res = []
    while n:
        n, r = -(n >> 1), n & 1
        res.append(str(r))
    return ''.join(res[::-1])
```

Time: O(log n) | Space: O(log n)

---

## 18. Prime Number of Set Bits in Binary Representation

**Description:** Count numbers in range [L,R] with prime number of set bits.

**Approach:** For each number count bits, check if count is prime.

```python
def countPrimeSetBits(left, right):
    primes = {2, 3, 5, 7, 11, 13, 17, 19}
    return sum(1 for x in range(left, right + 1) if bin(x).count('1') in primes)
```

Time: O(n * log n) | Space: O(1)

---

## 19. Number Complement

**Description:** Flip all bits of positive integer (leading zeros not flipped).

**Approach:** XOR with (1 << bit_length) - 1.

```python
def findComplement(num):
    mask = (1 << num.bit_length()) - 1
    return num ^ mask
```

Time: O(1) | Space: O(1)

---

## 20. Find the Difference

**Description:** String t is s with one random char added. Find the added char.

**Approach:** XOR all character codes; duplicates cancel.

```python
def findTheDifference(s, t):
    res = 0
    for c in s + t:
        res ^= ord(c)
    return chr(res)
```

Time: O(n) | Space: O(1)

---

## 21. Counting Bits

**Description:** Return array where ans[i] = number of 1s in binary of i.

**Approach:** DP: count[i] = count[i >> 1] + (i & 1).

```python
def countBits(n):
    res = [0] * (n + 1)
    for i in range(1, n + 1):
        res[i] = res[i >> 1] + (i & 1)
    return res
```

Time: O(n) | Space: O(1)

---

## 22. Sum of Two Integers

**Description:** Calculate a + b without using + or -.

**Approach:** XOR for sum, AND << 1 for carry; repeat until carry is 0.

```python
def getSum(a, b):
    mask = 0xFFFFFFFF
    while b & mask:
        carry = (a & b) << 1
        a = a ^ b
        b = carry
    return a & mask if b else a
```

Time: O(1) | Space: O(1)

---

## 23. Maximum Product of Word Lengths

**Description:** Max product of lengths of two words with no common letters.

**Approach:** Bitmask each word (a=bit0, b=bit1...); max len1*len2 where mask1 & mask2 == 0.

```python
def maxProduct(words):
    masks = [sum(1 << (ord(c) - 97) for c in set(w)) for w in words]
    res = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if not (masks[i] & masks[j]):
                res = max(res, len(words[i]) * len(words[j]))
    return res
```

Time: O(n^2) | Space: O(n)

---

## 24. Check if String Contains All Binary Codes of Size K

**Description:** Does string contain all 2^k binary codes of length k as substrings?

**Approach:** Rolling hash or bitmask; collect all k-length substrings as integers.

```python
def hasAllCodes(s, k):
    seen = set()
    for i in range(len(s) - k + 1):
        seen.add(s[i:i+k])
    return len(seen) == 2 ** k
```

Time: O(n * k) | Space: O(2^k)

---

## 25. Minimum Flips to Make a OR b Equal to c

**Description:** Flip minimum bits in a or b so that (a OR b) == c.

**Approach:** For each bit: if c has 0, both a and b must be 0; if c has 1, at least one of a,b must be 1.

```python
def minFlips(a, b, c):
    res = 0
    while a or b or c:
        if (c & 1) == 0:
            res += (a & 1) + (b & 1)
        else:
            res += 0 if (a & 1) or (b & 1) else 1
        a, b, c = a >> 1, b >> 1, c >> 1
    return res
```

Time: O(32) | Space: O(1)

---

## 26. Sort Integers by The Number of 1 Bits

**Description:** Sort array by popcount, then by value.

**Approach:** Use (popcount(x), x) as sort key.

```python
def sortByBits(arr):
    return sorted(arr, key=lambda x: (bin(x).count('1'), x))
```

Time: O(n log n) | Space: O(1)

---

## 27. Minimum Bit Flips to Convert Number

**Description:** Minimum bits to flip to convert start to goal.

**Approach:** XOR start and goal; count set bits.

```python
def minBitFlips(start, goal):
    return bin(start ^ goal).count('1')
```

Time: O(32) | Space: O(1)

---

## 28. XOR Queries of a Subarray

**Description:** Answer queries: XOR of arr[l..r].

**Approach:** Prefix XOR array; query = prefix[r+1] ^ prefix[l].

```python
def xorQueries(arr, queries):
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] ^ x)
    return [prefix[r+1] ^ prefix[l] for l, r in queries]
```

Time: O(n + q) | Space: O(n)
