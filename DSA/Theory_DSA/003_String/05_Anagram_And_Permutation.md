# Anagram and Permutation Theory and Implementations

## Theory

An anagram is a rearrangement of characters from one string to form another. Two strings are anagrams if they have the same character frequency. A permutation is any ordering of a set of elements.

## Check Anagram (Sort)

```python
def check_anagram_sort(a, b):
    return sorted(a) == sorted(b)
```

## Check Anagram (Frequency)

```python
def check_anagram_frequency(a, b):
    if len(a) != len(b):
        return False
    from collections import Counter
    return Counter(a) == Counter(b)

def check_anagram_frequency_manual(a, b):
    if len(a) != len(b):
        return False
    freq = [0] * 26
    for c in a:
        freq[ord(c) - ord("a")] += 1
    for c in b:
        freq[ord(c) - ord("a")] -= 1
        if freq[ord(c) - ord("a")] < 0:
            return False
    return True
```

## Valid Anagram

```python
def valid_anagram(s, t):
    return sorted(s) == sorted(t)
```

## Group Anagrams (Sort Key)

```python
def group_anagrams_sort(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

## Group Anagrams (Frequency Key)

```python
def group_anagrams_frequency(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        freq = [0] * 26
        for c in s:
            freq[ord(c) - ord("a")] += 1
        groups[tuple(freq)].append(s)
    return list(groups.values())
```

## Find All Anagrams (Sliding Window)

```python
def find_all_anagrams(s, p):
    if len(p) > len(s):
        return []
    from collections import Counter
    p_count = Counter(p)
    window_count = Counter(s[:len(p)])
    result = []
    if window_count == p_count:
        result.append(0)
    for i in range(len(p), len(s)):
        window_count[s[i]] = window_count.get(s[i], 0) + 1
        window_count[s[i - len(p)]] -= 1
        if window_count[s[i - len(p)]] == 0:
            del window_count[s[i - len(p)]]
        if window_count == p_count:
            result.append(i - len(p) + 1)
    return result
```

## Min Steps to Make Anagram

```python
def min_steps_anagram(s, t):
    from collections import Counter
    freq = Counter(s)
    for c in t:
        freq[c] -= 1
    return sum(abs(v) for v in freq.values()) // 2
```

## Min Swaps to Make Equal

```python
def min_swaps_equal(s1, s2):
    if s1 == s2:
        return 0
    diff = [(a, b) for a, b in zip(s1, s2) if a != b]
    if len(diff) % 2 != 0:
        return -1
    return len(diff) // 2
```

## Min Deletions for Anagram

```python
def min_deletions_anagram(s, t):
    from collections import Counter
    freq_s = Counter(s)
    freq_t = Counter(t)
    total = 0
    for c in set(s) | set(t):
        total += abs(freq_s.get(c, 0) - freq_t.get(c, 0))
    return total
```

## Rank of String Among Permutations

```python
def rank_of_string(s):
    from math import factorial
    n = len(s)
    rank = 1
    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            if s[j] < s[i]:
                count += 1
        rank += count * factorial(n - 1 - i)
    return rank
```

## Next Permutation

```python
def next_permutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i < 0:
        nums.reverse()
        return
    j = len(nums) - 1
    while nums[j] <= nums[i]:
        j -= 1
    nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1:] = reversed(nums[i + 1:])

def next_permutation_string(s):
    nums = list(s)
    next_permutation(nums)
    return "".join(nums)
```

## Check if Permutation of Palindrome

```python
def permutation_of_palindrome(s):
    from collections import Counter
    freq = Counter(s.replace(" ", "").lower())
    odd_count = sum(1 for v in freq.values() if v % 2 == 1)
    return odd_count <= 1
```

## Scramble String

```python
def is_scramble(s1, s2):
    if s1 == s2:
        return True
    if len(s1) != len(s2) or sorted(s1) != sorted(s2):
        return False
    n = len(s1)
    for i in range(1, n):
        if (is_scramble(s1[:i], s2[:i]) and is_scramble(s1[i:], s2[i:])) or \
           (is_scramble(s1[:i], s2[n-i:]) and is_scramble(s1[i:], s2[:n-i])):
            return True
    return False
```

## Smallest Window Containing All Chars

```python
def smallest_window_all_chars(s, t):
    from collections import Counter
    need = Counter(t)
    have = 0
    required = len(need)
    window = {}
    result = ""
    min_len = float("inf")
    left = 0
    for right, c in enumerate(s):
        window[c] = window.get(c, 0) + 1
        if c in need and window[c] == need[c]:
            have += 1
        while have == required:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = s[left:right + 1]
            window[s[left]] -= 1
            if s[left] in need and window[s[left]] < need[s[left]]:
                have -= 1
            left += 1
    return result
```

## Check Inclusion

```python
def check_inclusion(s1, s2):
    if len(s1) > len(s2):
        return False
    from collections import Counter
    s1_count = Counter(s1)
    window = Counter(s2[:len(s1)])
    if window == s1_count:
        return True
    for i in range(len(s1), len(s2)):
        window[s2[i]] = window.get(s2[i], 0) + 1
        window[s2[i - len(s1)]] -= 1
        if window[s2[i - len(s1)]] == 0:
            del window[s2[i - len(s1)]]
        if window == s1_count:
            return True
    return False
```

## Anagram Substring Search

```python
def anagram_substring_search(pattern, text):
    if len(pattern) > len(text):
        return []
    from collections import Counter
    p_count = Counter(pattern)
    window = Counter(text[:len(pattern)])
    result = []
    if window == p_count:
        result.append(0)
    for i in range(len(pattern), len(text)):
        window[text[i]] = window.get(text[i], 0) + 1
        window[text[i - len(pattern)]] -= 1
        if window[text[i - len(pattern)]] == 0:
            del window[text[i - len(pattern)]]
        if window == p_count:
            result.append(i - len(pattern) + 1)
    return result
```

## Count Anagram Occurrences

```python
def count_anagram_occurrences(text, pattern):
    return len(find_all_anagrams(text, pattern))

def find_all_anagrams(s, p):
    if len(p) > len(s):
        return []
    from collections import Counter
    p_count = Counter(p)
    window_count = Counter(s[:len(p)])
    result = []
    if window_count == p_count:
        result.append(0)
    for i in range(len(p), len(s)):
        window_count[s[i]] = window_count.get(s[i], 0) + 1
        window_count[s[i - len(p)]] -= 1
        if window_count[s[i - len(p)]] == 0:
            del window_count[s[i - len(p)]]
        if window_count == p_count:
            result.append(i - len(p) + 1)
    return result
```
