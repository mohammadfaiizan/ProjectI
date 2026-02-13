# Sliding Window Theory and Implementations

## Theory

Sliding window maintains a "window" of elements and slides it across the sequence. Two variants: fixed-size window and variable-size window. Used for substring problems with constraints (max distinct chars, min/max sum, etc.).

## Longest Without Repeating

```python
def longest_without_repeating(s):
    seen = {}
    start = 0
    max_len = 0
    for i, c in enumerate(s):
        if c in seen and seen[c] >= start:
            start = seen[c] + 1
        seen[c] = i
        max_len = max(max_len, i - start + 1)
    return max_len
```

## Min Window Substring

```python
def min_window_substring(s, t):
    from collections import Counter
    need = Counter(t)
    have = 0
    required = len(need)
    window = {}
    min_len = float("inf")
    result = ""
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

## Longest with At Most K Distinct

```python
def longest_k_distinct(s, k):
    if k == 0:
        return 0
    from collections import defaultdict
    freq = defaultdict(int)
    left = 0
    max_len = 0
    for right, c in enumerate(s):
        freq[c] += 1
        while len(freq) > k:
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Find All Anagrams

```python
def find_all_anagrams(s, p):
    if len(p) > len(s):
        return []
    from collections import Counter
    p_count = Counter(p)
    window = Counter(s[:len(p)])
    result = []
    if window == p_count:
        result.append(0)
    for i in range(len(p), len(s)):
        window[s[i]] = window.get(s[i], 0) + 1
        window[s[i - len(p)]] -= 1
        if window[s[i - len(p)]] == 0:
            del window[s[i - len(p)]]
        if window == p_count:
            result.append(i - len(p) + 1)
    return result
```

## Permutation in String

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

## Longest Repeating Char Replacement

```python
def character_replacement(s, k):
    from collections import Counter
    freq = Counter()
    max_freq = 0
    max_len = 0
    left = 0
    for right, c in enumerate(s):
        freq[c] += 1
        max_freq = max(max_freq, freq[c])
        if (right - left + 1) - max_freq > k:
            freq[s[left]] -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Fruit Into Baskets

```python
def total_fruit(fruits):
    from collections import defaultdict
    basket = defaultdict(int)
    left = 0
    max_fruits = 0
    for right, f in enumerate(fruits):
        basket[f] += 1
        while len(basket) > 2:
            basket[fruits[left]] -= 1
            if basket[fruits[left]] == 0:
                del basket[fruits[left]]
            left += 1
        max_fruits = max(max_fruits, right - left + 1)
    return max_fruits
```

## Smallest Window with All Chars

```python
def smallest_window_all_chars(s, t):
    from collections import Counter
    need = Counter(t)
    have = 0
    required = len(need)
    window = {}
    min_len = float("inf")
    result = ""
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

## Max Consecutive Ones

```python
def max_consecutive_ones(nums, k):
    left = 0
    zeros = 0
    max_len = 0
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Get Equal Substrings Within Budget

```python
def equal_substring(s, t, max_cost):
    left = 0
    cost = 0
    max_len = 0
    for right in range(len(s)):
        cost += abs(ord(s[right]) - ord(t[right]))
        while cost > max_cost:
            cost -= abs(ord(s[left]) - ord(t[left]))
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Max Vowels in Substring K

```python
def max_vowels(s, k):
    vowels = set("aeiou")
    count = sum(1 for c in s[:k] if c in vowels)
    max_count = count
    for i in range(k, len(s)):
        if s[i - k] in vowels:
            count -= 1
        if s[i] in vowels:
            count += 1
        max_count = max(max_count, count)
    return max_count
```

## Number of Substrings with Only 1s

```python
def num_substrings_ones(s):
    result = 0
    count = 0
    for c in s:
        if c == "1":
            count += 1
            result += count
        else:
            count = 0
    return result
```
