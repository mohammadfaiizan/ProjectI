# Advanced String Operations

## String Builder Pattern (List + Join)

```python
def string_builder(parts):
    result = []
    for part in parts:
        result.append(part)
    return "".join(result)

def string_builder_efficient():
    result = []
    for i in range(1000):
        result.append(str(i))
    return "".join(result)
```

## In-Place via List Conversion

```python
def inplace_via_list(s):
    chars = list(s)
    chars[0] = "X"
    return "".join(chars)

def inplace_swap(s, i, j):
    chars = list(s)
    chars[i], chars[j] = chars[j], chars[i]
    return "".join(chars)
```

## Tokenization and Parsing

```python
def tokenize(s, delimiters=" "):
    tokens = []
    current = []
    for c in s:
        if c in delimiters:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(c)
    if current:
        tokens.append("".join(current))
    return tokens

def parse_words(s):
    return s.split()
```

## Char Frequency Map with Ordering

```python
def frequency_ordered(s):
    from collections import OrderedDict
    freq = OrderedDict()
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return freq

def frequency_preserve_order(s):
    seen = {}
    order = []
    for c in s:
        if c not in seen:
            seen[c] = 0
            order.append(c)
        seen[c] += 1
    return {c: seen[c] for c in order}
```

## Lexicographic Comparison and Sorting

```python
def lexicographic_sort(strings):
    return sorted(strings)

def lexicographic_sort_key(strings, key=None):
    return sorted(strings, key=key)

def custom_sort_by_length_then_lex(strings):
    return sorted(strings, key=lambda s: (len(s), s))
```

## String Multiplication

```python
def string_multiply(s, n):
    return s * n

def repeat_pattern(s, n):
    return (s * ((n // len(s)) + 1))[:n]
```

## Interleave Two Strings

```python
def interleave(a, b):
    result = []
    i, j = 0, 0
    while i < len(a) or j < len(b):
        if i < len(a):
            result.append(a[i])
            i += 1
        if j < len(b):
            result.append(b[j])
            j += 1
    return "".join(result)

def interleave_equal_length(a, b):
    return "".join(a[i] + b[i] for i in range(len(a)))
```

## Remove Consecutive Duplicates

```python
def remove_consecutive_duplicates(s):
    if not s:
        return ""
    result = [s[0]]
    for i in range(1, len(s)):
        if s[i] != s[i - 1]:
            result.append(s[i])
    return "".join(result)
```

## Remove All Adjacent Duplicates

```python
def remove_adjacent_duplicates(s):
    stack = []
    for c in s:
        if stack and stack[-1] == c:
            stack.pop()
        else:
            stack.append(c)
    return "".join(stack)

def remove_adjacent_duplicates_k(s, k):
    stack = []
    for c in s:
        if stack and stack[-1][0] == c:
            stack[-1][1] += 1
            if stack[-1][1] == k:
                stack.pop()
        else:
            stack.append([c, 1])
    return "".join(c * cnt for c, cnt in stack)
```

## Zigzag Conversion

```python
def zigzag_convert(s, num_rows):
    if num_rows == 1 or num_rows >= len(s):
        return s
    rows = [""] * num_rows
    row, step = 0, 1
    for c in s:
        rows[row] += c
        row += step
        if row == 0 or row == num_rows - 1:
            step = -step
    return "".join(rows)
```

## String to Column Number and Back

```python
def column_to_number(s):
    result = 0
    for c in s:
        result = result * 26 + (ord(c) - ord("A") + 1)
    return result

def number_to_column(n):
    result = []
    while n:
        n -= 1
        result.append(chr(ord("A") + n % 26))
        n //= 26
    return "".join(reversed(result))
```

## Compare Version Numbers

```python
def compare_versions(v1, v2):
    parts1 = list(map(int, v1.split(".")))
    parts2 = list(map(int, v2.split(".")))
    n = max(len(parts1), len(parts2))
    for i in range(n):
        p1 = parts1[i] if i < len(parts1) else 0
        p2 = parts2[i] if i < len(parts2) else 0
        if p1 < p2:
            return -1
        if p1 > p2:
            return 1
    return 0
```

## Add Binary Strings

```python
def add_binary(a, b):
    carry = 0
    result = []
    i, j = len(a) - 1, len(b) - 1
    while i >= 0 or j >= 0 or carry:
        total = carry
        if i >= 0:
            total += int(a[i])
            i -= 1
        if j >= 0:
            total += int(b[j])
            j -= 1
        result.append(str(total % 2))
        carry = total // 2
    return "".join(reversed(result))
```

## Multiply Strings Digit by Digit

```python
def multiply_strings(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"
    m, n = len(num1), len(num2)
    result = [0] * (m + n)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            mul = int(num1[i]) * int(num2[j])
            p1, p2 = i + j, i + j + 1
            total = mul + result[p2]
            result[p2] = total % 10
            result[p1] += total // 10
    start = 0
    while start < len(result) and result[start] == 0:
        start += 1
    return "".join(str(x) for x in result[start:])
```

## Longest Common Prefix of Array

```python
def longest_common_prefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

def longest_common_prefix_vertical(strs):
    if not strs:
        return ""
    for i, c in enumerate(strs[0]):
        for s in strs[1:]:
            if i >= len(s) or s[i] != c:
                return strs[0][:i]
    return strs[0]
```

## Encode/Decode Strings (Serialize List)

```python
def encode_strings(strs):
    return "".join(f"{len(s)}#{s}" for s in strs)

def decode_strings(s):
    result = []
    i = 0
    while i < len(s):
        j = i
        while s[j] != "#":
            j += 1
        length = int(s[i:j])
        result.append(s[j + 1:j + 1 + length])
        i = j + 1 + length
    return result
```

## String Compression

```python
def string_compression(s):
    if not s:
        return ""
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(s[i - 1] + str(count))
            count = 1
    result.append(s[-1] + str(count))
    compressed = "".join(result)
    return compressed if len(compressed) < len(s) else s
```

## Count and Say

```python
def count_and_say(n):
    s = "1"
    for _ in range(n - 1):
        next_s = []
        i = 0
        while i < len(s):
            count = 1
            while i + 1 < len(s) and s[i + 1] == s[i]:
                count += 1
                i += 1
            next_s.append(str(count) + s[i])
            i += 1
        s = "".join(next_s)
    return s
```

## Reorganize String (No Two Adjacent Same)

```python
def reorganize_string(s):
    from collections import Counter
    from heapq import heappush, heappop
    freq = Counter(s)
    heap = [(-count, char) for char, count in freq.items()]
    import heapq
    heapq.heapify(heap)
    if -heap[0][0] > (len(s) + 1) // 2:
        return ""
    result = []
    prev = None
    while heap:
        count, char = heapq.heappop(heap)
        result.append(char)
        if prev:
            heapq.heappush(heap, prev)
        if count + 1 < 0:
            prev = (count + 1, char)
        else:
            prev = None
    return "".join(result)
```
