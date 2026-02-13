# Basic String Operations

## Create

```python
def create_empty():
    return ""

def create_from_literal():
    return "hello"

def create_from_chars(chars):
    return "".join(chars)

def create_repeated(char, n):
    return char * n
```

## Access Char by Index

```python
def access_by_index(s, i):
    return s[i]

def safe_access(s, i, default=None):
    try:
        return s[i]
    except IndexError:
        return default
```

## Traverse Forward

```python
def traverse_forward(s):
    for i in range(len(s)):
        print(s[i])

def traverse_forward_direct(s):
    for c in s:
        print(c)

def traverse_forward_enumerate(s):
    for i, c in enumerate(s):
        print(i, c)
```

## Traverse Backward

```python
def traverse_backward(s):
    for i in range(len(s) - 1, -1, -1):
        print(s[i])

def traverse_backward_reversed(s):
    for c in reversed(s):
        print(c)
```

## Find Length

```python
def find_length(s):
    return len(s)

def find_length_manual(s):
    count = 0
    for _ in s:
        count += 1
    return count
```

## Concatenate

```python
def concatenate(a, b):
    return a + b

def concatenate_multiple(*strings):
    return "".join(strings)
```

## Compare Lexicographic

```python
def compare_lexicographic(a, b):
    if a < b:
        return -1
    if a > b:
        return 1
    return 0

def compare_lexicographic_python(a, b):
    return (a > b) - (a < b)
```

## Check Equality

```python
def check_equality(a, b):
    return a == b

def check_equality_case_insensitive(a, b):
    return a.lower() == b.lower()
```

## Convert Case

```python
def to_upper(s):
    return s.upper()

def to_lower(s):
    return s.lower()

def to_title(s):
    return s.title()

def swap_case(s):
    return s.swapcase()
```

## Check Type

```python
def is_alpha(s):
    return s.isalpha()

def is_digit(s):
    return s.isdigit()

def is_alnum(s):
    return s.isalnum()

def is_space(s):
    return s.isspace()

def check_char_type(c):
    return {
        "alpha": c.isalpha(),
        "digit": c.isdigit(),
        "alnum": c.isalnum(),
        "space": c.isspace(),
    }
```

## Count Char Frequency

```python
def count_frequency_dict(s):
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return freq

def count_frequency_counter(s):
    from collections import Counter
    return dict(Counter(s))
```

## Find First Occurrence of Char

```python
def find_first(s, char):
    return s.find(char)

def find_first_index(s, char):
    try:
        return s.index(char)
    except ValueError:
        return -1

def find_first_manual(s, char):
    for i, c in enumerate(s):
        if c == char:
            return i
    return -1
```

## Find Last Occurrence of Char

```python
def find_last(s, char):
    return s.rfind(char)

def find_last_manual(s, char):
    for i in range(len(s) - 1, -1, -1):
        if s[i] == char:
            return i
    return -1
```

## Check Substring Exists

```python
def check_substring(s, sub):
    return sub in s

def check_substring_find(s, sub):
    return s.find(sub) != -1
```

## Extract Substring (Slicing)

```python
def extract_substring(s, start, end):
    return s[start:end]

def extract_from_start(s, n):
    return s[:n]

def extract_from_end(s, n):
    return s[-n:]

def extract_all_but_first(s):
    return s[1:]

def extract_all_but_last(s):
    return s[:-1]
```

## Split by Delimiter

```python
def split_by_delimiter(s, delim=" "):
    return s.split(delim)

def split_max_times(s, delim=" ", maxsplit=1):
    return s.split(delim, maxsplit)

def split_lines(s):
    return s.splitlines()
```

## Join List

```python
def join_list(lst, delim=""):
    return delim.join(lst)

def join_with_comma(lst):
    return ",".join(lst)
```

## Strip Whitespace

```python
def strip_whitespace(s):
    return s.strip()

def strip_left(s):
    return s.lstrip()

def strip_right(s):
    return s.rstrip()

def strip_chars(s, chars):
    return s.strip(chars)
```

## Replace Substring

```python
def replace_substring(s, old, new):
    return s.replace(old, new)

def replace_count(s, old, new, count=1):
    return s.replace(old, new, count)
```

## Reverse String

```python
def reverse_slicing(s):
    return s[::-1]

def reverse_two_pointers(s):
    chars = list(s)
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return "".join(chars)
```

## String to Int (atoi)

```python
def atoi(s):
    s = s.strip()
    if not s:
        return 0
    sign = 1
    if s[0] == "-":
        sign = -1
        s = s[1:]
    elif s[0] == "+":
        s = s[1:]
    result = 0
    for c in s:
        if not c.isdigit():
            break
        result = result * 10 + int(c)
    return max(-2**31, min(2**31 - 1, sign * result))
```

## Int to String

```python
def int_to_string(n):
    return str(n)

def int_to_string_manual(n):
    if n == 0:
        return "0"
    if n < 0:
        return "-" + int_to_string_manual(-n)
    result = []
    while n:
        result.append(chr(ord("0") + n % 10))
        n //= 10
    return "".join(reversed(result))
```

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

## Check Rotations

```python
def check_rotations(s1, s2):
    if len(s1) != len(s2):
        return False
    return s2 in (s1 + s1)

def check_rotations_manual(s1, s2):
    if len(s1) != len(s2):
        return False
    doubled = s1 + s1
    for i in range(len(s1)):
        if doubled[i:i + len(s1)] == s2:
            return True
    return False
```
