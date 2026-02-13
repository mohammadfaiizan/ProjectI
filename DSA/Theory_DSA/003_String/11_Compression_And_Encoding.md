# Compression and Encoding Theory and Implementations

## Theory

Compression reduces string size by exploiting redundancy. Encoding transforms data for storage or transmission. Common schemes: run-length, Huffman, length-prefixed.

## Run-Length Encoding (Encode)

```python
def run_length_encode(s):
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
    return "".join(result)
```

## Run-Length Encoding (Decode)

```python
def run_length_decode(s):
    result = []
    i = 0
    while i < len(s):
        char = s[i]
        i += 1
        num = ""
        while i < len(s) and s[i].isdigit():
            num += s[i]
            i += 1
        result.append(char * int(num) if num else char)
    return "".join(result)
```

## String Compression (aabccc to a2b1c3)

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

## Decode String (Nested Brackets)

```python
def decode_string(s):
    stack = []
    current_num = ""
    current_str = ""
    for c in s:
        if c.isdigit():
            current_num += c
        elif c == "[":
            stack.append((current_str, int(current_num) if current_num else 1))
            current_str = ""
            current_num = ""
        elif c == "]":
            prev_str, repeat = stack.pop()
            current_str = prev_str + current_str * repeat
        else:
            current_str += c
    return current_str
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

## Encode/Decode Strings (Length-Prefixed)

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

## Caesar Cipher

```python
def caesar_encrypt(s, shift):
    result = []
    for c in s:
        if c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            result.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            result.append(c)
    return "".join(result)

def caesar_decrypt(s, shift):
    return caesar_encrypt(s, -shift)
```

## Huffman Coding Overview

Build a binary tree from character frequencies. Frequent chars get shorter codes. Encode by traversing tree; decode by walking tree with bit stream.

```python
from heapq import heappush, heappop

class Node:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(s):
    from collections import Counter
    freq = Counter(s)
    heap = [Node(char=c, freq=f) for c, f in freq.items()]
    import heapq
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    return heapq.heappop(heap) if heap else None

def build_codes(root, path="", codes=None):
    if codes is None:
        codes = {}
    if root.char is not None:
        codes[root.char] = path or "0"
        return
    build_codes(root.left, path + "0", codes)
    build_codes(root.right, path + "1", codes)
    return codes

def huffman_encode(s, codes):
    return "".join(codes[c] for c in s)

def huffman_decode(encoded, root):
    if not root:
        return ""
    result = []
    node = root
    for bit in encoded:
        node = node.left if bit == "0" else node.right
        if node.char is not None:
            result.append(node.char)
            node = root
    return "".join(result)
```
