# Medium String Problems

## 01. Longest Substring Without Repeating Characters

**Description**: Find length of longest substring with all unique chars.
**Approach**: Sliding window with hash map of char to last index.

```python
def length_of_longest_substring(s):
    seen = {}
    start = best = 0
    for i, c in enumerate(s):
        if c in seen and seen[c] >= start:
            start = seen[c] + 1
        seen[c] = i
        best = max(best, i - start + 1)
    return best
```

Time: O(n) | Space: O(min(n, charset))

---

## 02. Longest Palindromic Substring

**Description**: Find longest palindrome substring.
**Approach**: Expand around center for each index (odd and even length), or Manacher O(n).

```python
def longest_palindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]
    best = ""
    for i in range(len(s)):
        best = max(expand(i, i), expand(i, i + 1), best, key=len)
    return best
```

Time: O(n^2) | Space: O(1)

---

## 03. Zigzag Conversion

**Description**: Write string in zigzag pattern, read row by row.
**Approach**: Simulate row indices: 0,1,...,numRows-1,numRows-2,...,0.

```python
def convert_zigzag(s, num_rows):
    if num_rows == 1:
        return s
    rows = [""] * num_rows
    r, step = 0, 1
    for c in s:
        rows[r] += c
        r += step
        if r == 0 or r == num_rows - 1:
            step *= -1
    return "".join(rows)
```

Time: O(n) | Space: O(n)

---

## 04. String to Integer (atoi)

**Description**: Parse integer from string with overflow.
**Approach**: Strip, sign, accumulate digits, clamp to 32-bit.

```python
def my_atoi(s):
    s = s.strip()
    if not s:
        return 0
    sign = -1 if s[0] == '-' else 1
    if s[0] in '+-':
        s = s[1:]
    res = 0
    for c in s:
        if not c.isdigit():
            break
        res = res * 10 + int(c)
    res *= sign
    return max(-2**31, min(2**31 - 1, res))
```

Time: O(n) | Space: O(1)

---

## 05. Letter Combinations of Phone Number

**Description**: All letter combos for digit string (2=abc, 3=def, etc).
**Approach**: Backtracking or iterative BFS.

```python
def letter_combinations(digits):
    if not digits:
        return []
    d = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
    out = []
    def bt(i, path):
        if i == len(digits):
            out.append("".join(path))
            return
        for c in d[digits[i]]:
            path.append(c)
            bt(i + 1, path)
            path.pop()
    bt(0, [])
    return out
```

Time: O(4^n) | Space: O(n)

---

## 06. Generate Parentheses

**Description**: Generate all valid n pairs of parentheses.
**Approach**: Backtrack: add "(" if open < n, add ")" if close < open.

```python
def generate_parenthesis(n):
    out = []
    def bt(s, open_c, close_c):
        if len(s) == 2 * n:
            out.append(s)
            return
        if open_c < n:
            bt(s + "(", open_c + 1, close_c)
        if close_c < open_c:
            bt(s + ")", open_c, close_c + 1)
    bt("", 0, 0)
    return out
```

Time: O(4^n / sqrt(n)) | Space: O(n)

---

## 07. Group Anagrams

**Description**: Group strings that are anagrams.
**Approach**: Use sorted string or frequency tuple as key in defaultdict.

```python
def group_anagrams(strs):
    from collections import defaultdict
    d = defaultdict(list)
    for s in strs:
        d[tuple(sorted(s))].append(s)
    return list(d.values())
```

Time: O(n * k log k) | Space: O(n)

---

## 08. Decode Ways

**Description**: Number of ways to decode digit string (1=A,...,26=Z).
**Approach**: DP: dp[i] = ways for s[:i], consider 1-digit and 2-digit.

```python
def num_decodings(s):
    if not s or s[0] == '0':
        return 0
    prev, cur = 1, 1
    for i in range(1, len(s)):
        tmp = 0
        if s[i] != '0':
            tmp = cur
        if 10 <= int(s[i-1:i+1]) <= 26:
            tmp += prev
        prev, cur = cur, tmp
    return cur
```

Time: O(n) | Space: O(1)

---

## 09. Word Break

**Description**: Can string be segmented into dictionary words?
**Approach**: DP: dp[i] = True if s[:i] can be broken.

```python
def word_break(s, word_dict):
    wd = set(word_dict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in wd:
                dp[i] = True
                break
    return dp[-1]
```

Time: O(n^2 * m) | Space: O(n)

---

## 10. Longest Repeating Character Replacement

**Description**: Longest substring with same char after at most k replacements.
**Approach**: Sliding window, track max freq in window, shrink when (len - max_freq) > k.

```python
def character_replacement(s, k):
    from collections import Counter
    cnt = Counter()
    best = mx_freq = 0
    l = 0
    for r, c in enumerate(s):
        cnt[c] += 1
        mx_freq = max(mx_freq, cnt[c])
        while (r - l + 1) - mx_freq > k:
            cnt[s[l]] -= 1
            l += 1
        best = max(best, r - l + 1)
    return best
```

Time: O(n) | Space: O(1)

---

## 11. Find All Anagrams in String

**Description**: Starting indices where anagram of p exists in s.
**Approach**: Sliding window of len(p), compare frequency with p.

```python
def find_anagrams(s, p):
    from collections import Counter
    if len(p) > len(s):
        return []
    pc = Counter(p)
    sc = Counter(s[:len(p)])
    out = [0] if pc == sc else []
    for i in range(len(p), len(s)):
        sc[s[i]] += 1
        sc[s[i - len(p)]] -= 1
        if sc[s[i - len(p)]] == 0:
            del sc[s[i - len(p)]]
        if pc == sc:
            out.append(i - len(p) + 1)
    return out
```

Time: O(n) | Space: O(1)

---

## 12. Permutation in String

**Description**: Does s2 contain permutation of s1?
**Approach**: Sliding window, check if window freq equals s1 freq.

```python
def check_inclusion(s1, s2):
    from collections import Counter
    if len(s1) > len(s2):
        return False
    c1 = Counter(s1)
    c2 = Counter(s2[:len(s1)])
    if c1 == c2:
        return True
    for i in range(len(s1), len(s2)):
        c2[s2[i]] += 1
        c2[s2[i - len(s1)]] -= 1
        if c2[s2[i - len(s1)]] == 0:
            del c2[s2[i - len(s1)]]
        if c1 == c2:
            return True
    return False
```

Time: O(n) | Space: O(1)

---

## 13. Minimum Window Substring

**Description**: Smallest substring of s containing all chars of t.
**Approach**: Sliding window, expand until valid, shrink from left, track min.

```python
def min_window(s, t):
    from collections import Counter
    need = Counter(t)
    have = 0
    need_cnt = len(need)
    start, length = 0, float('inf')
    l = 0
    for r, c in enumerate(s):
        if c in need:
            need[c] -= 1
            if need[c] == 0:
                have += 1
        while have == need_cnt:
            if r - l + 1 < length:
                start, length = l, r - l + 1
            if s[l] in need:
                need[s[l]] += 1
                if need[s[l]] > 0:
                    have -= 1
            l += 1
    return s[start:start + length] if length != float('inf') else ""
```

Time: O(n + m) | Space: O(m)

---

## 14. Substring with Concatenation of All Words

**Description**: Find indices where concatenation of all words appears.
**Approach**: Sliding window over possible starts, check each word in window.

```python
def find_substring(s, words):
    from collections import Counter
    if not words:
        return []
    wlen, nw = len(words[0]), len(words)
    target = Counter(words)
    out = []
    for start in range(wlen):
        seen = Counter()
        cnt = 0
        for i in range(start, len(s) - wlen + 1, wlen):
            w = s[i:i + wlen]
            if w in target:
                seen[w] += 1
                cnt += 1
                while seen[w] > target[w]:
                    first = s[start:start + wlen]
                    seen[first] -= 1
                    cnt -= 1
                    start += wlen
                if cnt == nw:
                    out.append(start)
            else:
                seen.clear()
                cnt = 0
                start = i + wlen
    return out
```

Time: O(n * wlen) | Space: O(nw)

---

## 15. Longest Palindromic Subsequence

**Description**: Length of longest palindromic subsequence.
**Approach**: DP: dp[i][j] = LPS of s[i:j+1], recurse on ends.

```python
def longest_palindrome_subseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            if s[i] == s[j]:
                dp[i][j] = 2 + (dp[i + 1][j - 1] if L > 2 else 0)
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

Time: O(n^2) | Space: O(n^2)

---

## 16. Palindromic Substrings

**Description**: Count all palindromic substrings.
**Approach**: Expand around center for each index (odd and even).

```python
def count_substrings(s):
    def expand(l, r):
        cnt = 0
        while l >= 0 and r < len(s) and s[l] == s[r]:
            cnt += 1
            l -= 1
            r += 1
        return cnt
    total = 0
    for i in range(len(s)):
        total += expand(i, i) + expand(i, i + 1)
    return total
```

Time: O(n^2) | Space: O(1)

---

## 17. Encode and Decode Strings

**Description**: Serialize list of strings for transmission.
**Approach**: Length-prefixed: "4#word" format, parse by reading length then chars.

```python
def encode(strs):
    return "".join(f"{len(s)}#{s}" for s in strs)

def decode(s):
    out, i = [], 0
    while i < len(s):
        j = s.index('#', i)
        ln = int(s[i:j])
        out.append(s[j + 1:j + 1 + ln])
        i = j + 1 + ln
    return out
```

Time: O(n) | Space: O(n)

---

## 18. Reorganize String

**Description**: Reorder so no two adjacent same.
**Approach**: Max-heap by frequency, alternate most frequent with others.

```python
def reorganize_string(s):
    from collections import Counter
    import heapq
    c = Counter(s)
    if max(c.values()) > (len(s) + 1) // 2:
        return ""
    h = [(-v, k) for k, v in c.items()]
    heapq.heapify(h)
    res = []
    prev = None
    while h:
        v, k = heapq.heappop(h)
        if prev and prev == k:
            v2, k2 = heapq.heappop(h)
            res.append(k2)
            if v2 + 1 < 0:
                heapq.heappush(h, (v2 + 1, k2))
            heapq.heappush(h, (v, k))
            prev = k2
        else:
            res.append(k)
            if v + 1 < 0:
                heapq.heappush(h, (v + 1, k))
            prev = k
    return "".join(res)
```

Time: O(n log k) | Space: O(k)

---

## 19. Compare Version Numbers

**Description**: Compare two version strings (1.0.1 vs 1.0.0).
**Approach**: Split by ".", pad with zeros, compare numerically.

```python
def compare_version(v1, v2):
    a = list(map(int, v1.split('.')))
    b = list(map(int, v2.split('.')))
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else 0
        y = b[i] if i < len(b) else 0
        if x > y:
            return 1
        if x < y:
            return -1
    return 0
```

Time: O(n) | Space: O(n)

---

## 20. Multiply Strings

**Description**: Multiply two numbers as strings.
**Approach**: Digit-by-digit multiplication, store in array, handle carries.

```python
def multiply(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"
    m, n = len(num1), len(num2)
    res = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            p = int(num1[i]) * int(num2[j]) + res[i + j + 1]
            res[i + j + 1] = p % 10
            res[i + j] += p // 10
    return "".join(map(str, res)).lstrip('0')
```

Time: O(m * n) | Space: O(m + n)

---

## 21. Simplify Path

**Description**: Simplify Unix path (remove . and .., collapse slashes).
**Approach**: Split by "/", use stack: push for dir, pop for "..", ignore ".".

```python
def simplify_path(path):
    st = []
    for part in path.split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            if st:
                st.pop()
        else:
            st.append(part)
    return '/' + '/'.join(st)
```

Time: O(n) | Space: O(n)

---

## 22. Basic Calculator II

**Description**: Evaluate expression with +, -, *, /.
**Approach**: Parse and evaluate, handle * and / first (two-pass or stack).

```python
def calculate(s):
    s = s.replace(' ', '')
    num, st, op = 0, [], '+'
    for i, c in enumerate(s + '+'):
        if c.isdigit():
            num = num * 10 + int(c)
        elif c in '+-*/':
            if op == '+':
                st.append(num)
            elif op == '-':
                st.append(-num)
            elif op == '*':
                st.append(st.pop() * num)
            else:
                st.append(int(st.pop() / num))
            num, op = 0, c
    return sum(st)
```

Time: O(n) | Space: O(n)

---

## 23. Restore IP Addresses

**Description**: All valid IP addresses from string.
**Approach**: Backtrack: place 3 dots, check each segment 0-255 and no leading zeros.

```python
def restore_ip_addresses(s):
    out = []
    def bt(start, path):
        if len(path) == 4 and start == len(s):
            out.append(".".join(path))
            return
        if len(path) >= 4 or start >= len(s):
            return
        for i in range(1, 4):
            seg = s[start:start + i]
            if (seg[0] == '0' and len(seg) > 1) or int(seg) > 255:
                continue
            path.append(seg)
            bt(start + i, path)
            path.pop()
    bt(0, [])
    return out
```

Time: O(1) | Space: O(1)

---

## 24. Word Search

**Description**: Does grid contain word (adjacent cells)?
**Approach**: DFS from each cell, backtrack with visited set.

```python
def exist(board, word):
    m, n = len(board), len(board[0])
    def dfs(i, j, k):
        if k == len(word):
            return True
        if i < 0 or i >= m or j < 0 or j >= n or board[i][j] != word[k]:
            return False
        tmp, board[i][j] = board[i][j], '#'
        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
            if dfs(i + di, j + dj, k + 1):
                return True
        board[i][j] = tmp
        return False
    return any(dfs(i, j, 0) for i in range(m) for j in range(n))
```

Time: O(m * n * 4^L) | Space: O(L)

---

## 25. Implement Trie

**Description**: Implement prefix tree (insert, search, startsWith).
**Approach**: Node with children dict, leaf marker for complete words.

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False
    def insert(self, word):
        node = self
        for c in word:
            if c not in node.children:
                node.children[c] = Trie()
            node = node.children[c]
        node.is_end = True
    def search(self, word):
        node = self
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.is_end
    def starts_with(self, prefix):
        node = self
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True
```

Time: O(n) per op | Space: O(n * L)
