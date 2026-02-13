# Subsequence and Substring Theory and Implementations

## Theory

**Substring**: Contiguous sequence of characters. For s of length n, there are n*(n+1)/2 substrings.

**Subsequence**: Characters in order but not necessarily contiguous. For s of length n, there are 2^n subsequences (each char included or not).

## Is Subsequence Check

```python
def is_subsequence(s, t):
    i = j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)
```

## Number of Distinct Subsequences

```python
def distinct_subsequences(s):
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    last = {}
    for i in range(1, n + 1):
        dp[i] = 2 * dp[i - 1]
        if s[i - 1] in last:
            dp[i] -= dp[last[s[i - 1]]]
        last[s[i - 1]] = i - 1
    return dp[n]
```

## Longest Common Subsequence (DP)

```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

## Longest Common Substring (DP)

```python
def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_pos = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0
    return s1[end_pos - max_len:end_pos]
```

## Longest Repeating Substring

```python
def longest_repeating_substring(s):
    n = len(s)
    for length in range(n - 1, 0, -1):
        seen = set()
        for i in range(n - length + 1):
            sub = s[i:i + length]
            if sub in seen:
                return sub
            seen.add(sub)
    return ""

def longest_repeating_substring_binary_search(s):
    def has_repeat(length):
        seen = set()
        for i in range(len(s) - length + 1):
            sub = s[i:i + length]
            if sub in seen:
                return True
            seen.add(sub)
        return False

    left, right = 1, len(s) - 1
    result = 0
    while left <= right:
        mid = (left + right) // 2
        if has_repeat(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result
```

## Longest Substring Without Repeating Chars

```python
def longest_substring_no_repeat(s):
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

## Longest with At Most 2 Distinct

```python
def longest_2_distinct(s):
    return longest_k_distinct(s, 2)
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

## Distinct Subsequences Count

```python
def num_distinct_subsequences(s, t):
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j]
            if s[i - 1] == t[j - 1]:
                dp[i][j] += dp[i - 1][j - 1]
    return dp[m][n]
```

## Longest Palindromic Subsequence

```python
def longest_palindromic_subsequence(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = 2 + dp[i + 1][j - 1] if length > 2 else 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

## Shortest Common Supersequence

```python
def shortest_common_supersequence(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    return m + n - lcs_len

def scs_string(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    result = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and str1[i - 1] == str2[j - 1]:
            result.append(str1[i - 1])
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            result.append(str2[j - 1])
            j -= 1
        else:
            result.append(str1[i - 1])
            i -= 1
    return "".join(reversed(result))
```

## Longest Repeating Character Replacement

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

## Count Unique Chars of All Substrings

```python
def unique_letter_string(s):
    n = len(s)
    last = {}
    prev = [0] * n
    for i, c in enumerate(s):
        prev[i] = last.get(c, -1)
        last[c] = i
    last = {}
    next_pos = [n] * n
    for i in range(n - 1, -1, -1):
        next_pos[i] = last.get(s[i], n)
        last[s[i]] = i
    result = 0
    for i in range(n):
        result += (i - prev[i]) * (next_pos[i] - i)
    return result
```

## Substrings Containing All Three Chars

```python
def count_substrings_all_three(s):
    result = 0
    last = {0: -1, 1: -1, 2: -1}
    for i, c in enumerate(s):
        if c in "abc":
            last[ord(c) - ord("a")] = i
        result += 1 + min(last[0], last[1], last[2])
    return result
```

## Substring Concatenation of All Words

```python
def find_substring(s, words):
    if not words:
        return []
    from collections import Counter
    word_len = len(words[0])
    total_len = word_len * len(words)
    word_count = Counter(words)
    result = []
    for i in range(len(s) - total_len + 1):
        seen = {}
        j = 0
        while j < len(words):
            word = s[i + j * word_len:i + (j + 1) * word_len]
            if word not in word_count:
                break
            seen[word] = seen.get(word, 0) + 1
            if seen[word] > word_count[word]:
                break
            j += 1
        if j == len(words):
            result.append(i)
    return result
```
