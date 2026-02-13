# XOR Techniques

## XOR Properties

- a ^ a = 0 (self-inverse)
- a ^ 0 = a (identity)
- Commutative: a ^ b = b ^ a
- Associative: (a ^ b) ^ c = a ^ (b ^ c)

## Single Number (One Unique, Rest Twice)

```python
def single_number(nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result
```

## Single Number II (One Once, Rest Three Times - Bit Counting)

```python
def single_number_ii(nums: list[int]) -> int:
    ones = 0
    twos = 0
    for num in nums:
        twos |= ones & num
        ones ^= num
        threes = ones & twos
        ones &= ~threes
        twos &= ~threes
    return ones

def single_number_ii_bit_count(nums: list[int]) -> int:
    result = 0
    for i in range(32):
        count = 0
        for num in nums:
            if (num >> i) & 1:
                count += 1
        if count % 3:
            result |= 1 << i
    return result if result < 2**31 else result - 2**32
```

## Single Number III (Two Unique, Rest Twice - XOR + Partition)

```python
def single_number_iii(nums: list[int]) -> list[int]:
    xor_all = 0
    for num in nums:
        xor_all ^= num
    rightmost = xor_all & -xor_all
    a = b = 0
    for num in nums:
        if num & rightmost:
            a ^= num
        else:
            b ^= num
    return [a, b]
```

## Missing Number (XOR 0..n with Array)

```python
def missing_number(nums: list[int]) -> int:
    n = len(nums)
    xor_all = 0
    for i in range(n + 1):
        xor_all ^= i
    for num in nums:
        xor_all ^= num
    return xor_all
```

## Find Duplicate Number (XOR Approach)

```python
def find_duplicate_xor(nums: list[int]) -> int:
    n = len(nums) - 1
    xor_all = 0
    for i in range(1, n + 1):
        xor_all ^= i
    for num in nums:
        xor_all ^= num
    return xor_all
```

## XOR of Range [L, R] (Using 1 to n Pattern)

```python
def xor_1_to_n(n: int) -> int:
    if n % 4 == 0:
        return n
    if n % 4 == 1:
        return 1
    if n % 4 == 2:
        return n + 1
    return 0

def xor_range(l: int, r: int) -> int:
    return xor_1_to_n(r) ^ xor_1_to_n(l - 1)
```

## Find Two Missing Numbers

```python
def find_two_missing(nums: list[int], n: int) -> list[int]:
    xor_all = 0
    for i in range(1, n + 1):
        xor_all ^= i
    for num in nums:
        xor_all ^= num
    rightmost = xor_all & -xor_all
    a = b = 0
    for i in range(1, n + 1):
        if i & rightmost:
            a ^= i
        else:
            b ^= i
    for num in nums:
        if num & rightmost:
            a ^= num
        else:
            b ^= num
    return [a, b]
```

## Maximum XOR of Two Numbers (Trie Approach)

```python
class TrieNode:
    def __init__(self):
        self.children = {}

def build_trie(nums: list[int]) -> TrieNode:
    root = TrieNode()
    for num in nums:
        node = root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]
    return root

def find_max_xor(nums: list[int]) -> int:
    if not nums:
        return 0
    root = build_trie(nums)
    max_xor = 0
    for num in nums:
        node = root
        curr_xor = 0
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            toggled = 1 - bit
            if toggled in node.children:
                curr_xor |= 1 << i
                node = node.children[toggled]
            else:
                node = node.children[bit]
        max_xor = max(max_xor, curr_xor)
    return max_xor
```

## Minimum XOR Sum of Two Arrays (Bitmask DP Note)

State: dp[mask] = minimum XOR sum when we have assigned first popcount(mask) elements of arr1 to indices in mask. O(n^2 * 2^n).

```python
def minimum_xor_sum(arr1: list[int], arr2: list[int]) -> int:
    n = len(arr1)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    for mask in range(1 << n):
        j = bin(mask).count('1')
        if j >= n:
            continue
        for i in range(n):
            if (mask >> i) & 1:
                continue
            new_mask = mask | (1 << i)
            dp[new_mask] = min(dp[new_mask], dp[mask] + (arr1[j] ^ arr2[i]))
    return dp[(1 << n) - 1]
```

## Find XOR of All Subsets

For array of n elements, each element appears in 2^(n-1) subsets. So total XOR = 0 if n > 1, else arr[0].

```python
def xor_of_all_subsets(nums: list[int]) -> int:
    if len(nums) == 1:
        return nums[0]
    return 0
```

## Total Hamming Distance

```python
def total_hamming_distance(nums: list[int]) -> int:
    n = len(nums)
    total = 0
    for i in range(32):
        count = sum((num >> i) & 1 for num in nums)
        total += count * (n - count)
    return total
```

## Hamming Distance Two Numbers

```python
def hamming_distance(a: int, b: int) -> int:
    xor_val = a ^ b
    return bin(xor_val).count('1')

def hamming_distance_v2(a: int, b: int) -> int:
    xor_val = a ^ b
    count = 0
    while xor_val:
        count += 1
        xor_val &= xor_val - 1
    return count
```

## Complement of Number

```python
def find_complement(num: int) -> int:
    mask = (1 << num.bit_length()) - 1
    return num ^ mask
```

## Decode XORed Array

```python
def decode(encoded: list[int], first: int) -> list[int]:
    arr = [first]
    for x in encoded:
        arr.append(arr[-1] ^ x)
    return arr
```

## Decode XORed Permutation

```python
def decode_permutation(encoded: list[int]) -> list[int]:
    n = len(encoded) + 1
    total_xor = 0
    for i in range(1, n + 1):
        total_xor ^= i
    odd_xor = 0
    for i in range(1, len(encoded), 2):
        odd_xor ^= encoded[i]
    first = total_xor ^ odd_xor
    arr = [first]
    for x in encoded:
        arr.append(arr[-1] ^ x)
    return arr
```

## XOR Queries of Subarray

```python
def xor_queries(arr: list[int], queries: list[list[int]]) -> list[int]:
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] ^ x)
    return [prefix[r + 1] ^ prefix[l] for l, r in queries]
```
