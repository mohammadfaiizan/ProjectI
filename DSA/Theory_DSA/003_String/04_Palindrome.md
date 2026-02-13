# Palindrome Theory and Implementations

## Theory

A palindrome reads the same forward and backward. Formally: s[i] == s[n-1-i] for all valid i. Palindromes can be odd-length (center char) or even-length (no center).

## Check Palindrome

```python
def check_palindrome(s):
    return s == s[::-1]

def check_palindrome_two_pointers(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

## Check Ignoring Non-Alphanumeric

```python
def check_palindrome_alphanumeric(s):
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]

def check_palindrome_alphanumeric_two_pointers(s):
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True
```

## Valid Palindrome II (One Deletion)

```python
def valid_palindrome_ii(s):
    def is_palindrome(left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True

    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1
        right -= 1
    return True
```

## Longest Palindromic Substring - Brute Force

```python
def longest_palindromic_substring_brute(s):
    n = len(s)
    best = ""
    for i in range(n):
        for j in range(i, n):
            sub = s[i:j + 1]
            if sub == sub[::-1] and len(sub) > len(best):
                best = sub
    return best
```

## Longest Palindromic Substring - Expand Around Center

```python
def longest_palindromic_substring_expand(s):
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]

    if not s:
        return ""
    result = ""
    for i in range(len(s)):
        odd = expand(i, i)
        even = expand(i, i + 1)
        result = max(result, odd, even, key=len)
    return result
```

## Longest Palindromic Substring - Manacher O(n)

```python
def longest_palindromic_substring_manacher(s):
    t = "#" + "#".join(s) + "#"
    n = len(t)
    p = [0] * n
    c, r = 0, 0
    max_len, center = 0, 0
    for i in range(n):
        if i < r:
            mirror = 2 * c - i
            p[i] = min(r - i, p[mirror])
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]
        if p[i] > max_len:
            max_len, center = p[i], i
    start = (center - max_len) // 2
    return s[start:start + max_len]
```

## Longest Palindromic Subsequence (DP)

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

## Count Palindromic Substrings

```python
def count_palindromic_substrings(s):
    def expand(left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count

    total = 0
    for i in range(len(s)):
        total += expand(i, i)
        total += expand(i, i + 1)
    return total
```

## Palindrome Partitioning (All Partitions Backtracking)

```python
def palindrome_partitioning(s):
    def is_palindrome(sub):
        return sub == sub[::-1]

    def backtrack(start, path, result):
        if start == len(s):
            result.append(path[:])
            return
        for end in range(start + 1, len(s) + 1):
            sub = s[start:end]
            if is_palindrome(sub):
                path.append(sub)
                backtrack(end, path, result)
                path.pop()

    result = []
    backtrack(0, [], result)
    return result
```

## Palindrome Partitioning II (Min Cuts DP)

```python
def palindrome_partitioning_ii(s):
    n = len(s)
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length == 2 or is_pal[i + 1][j - 1]):
                is_pal[i][j] = True

    cuts = [float("inf")] * (n + 1)
    cuts[0] = -1
    for j in range(1, n + 1):
        for i in range(j):
            if is_pal[i][j - 1]:
                cuts[j] = min(cuts[j], cuts[i] + 1)
    return cuts[n]
```

## Palindrome Pairs

```python
def palindrome_pairs(words):
    def is_palindrome(s):
        return s == s[::-1]

    word_index = {w: i for i, w in enumerate(words)}
    result = []
    for i, word in enumerate(words):
        for j in range(len(word) + 1):
            prefix, suffix = word[:j], word[j:]
            if is_palindrome(prefix):
                rev = suffix[::-1]
                if rev in word_index and word_index[rev] != i:
                    result.append([word_index[rev], i])
            if j != len(word) and is_palindrome(suffix):
                rev = prefix[::-1]
                if rev in word_index and word_index[rev] != i:
                    result.append([i, word_index[rev]])
    return result
```

## Shortest Palindrome (Prepend)

```python
def shortest_palindrome(s):
    t = s + "#" + s[::-1]
    n = len(t)
    lps = [0] * n
    for i in range(1, n):
        j = lps[i - 1]
        while j > 0 and t[i] != t[j]:
            j = lps[j - 1]
        if t[i] == t[j]:
            j += 1
        lps[i] = j
    return s[lps[-1]:][::-1] + s
```

## Min Insertions to Make Palindrome

```python
def min_insertions_palindrome(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

## Check if Rearrangement Can Form Palindrome

```python
def can_form_palindrome(s):
    from collections import Counter
    freq = Counter(s)
    odd_count = sum(1 for v in freq.values() if v % 2 == 1)
    return odd_count <= 1
```

## Longest Palindrome from Chars

```python
def longest_palindrome_from_chars(s):
    from collections import Counter
    freq = Counter(s)
    length = 0
    has_odd = False
    for count in freq.values():
        length += (count // 2) * 2
        if count % 2 == 1:
            has_odd = True
    return length + (1 if has_odd else 0)
```

## Break a Palindrome

```python
def break_palindrome(s):
    if len(s) == 1:
        return ""
    s = list(s)
    for i in range(len(s) // 2):
        if s[i] != "a":
            s[i] = "a"
            return "".join(s)
    s[-1] = "b"
    return "".join(s)
```

## K-Palindrome

```python
def k_palindrome(s, k):
    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    rev = s[::-1]
    return len(s) - lcs(s, rev) <= k
```
