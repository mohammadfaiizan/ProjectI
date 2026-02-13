# Greedy - Huffman Coding

## Huffman Coding Concept (Variable Length Prefix Codes)

Huffman coding assigns variable-length binary codes to characters. Frequent characters get shorter codes. Codes are prefix-free: no code is a prefix of another, enabling unique decoding.

Goal: minimize expected code length = sum over chars of (freq(char) * len(code(char))).

## Build Huffman Tree (Min-Heap Merge Two Smallest)

Repeatedly merge two nodes with smallest frequency. The result is a binary tree where leaves are characters and path from root gives the code.

```python
import heapq
from collections import Counter

class Node:
    def __init__(self, freq, char=None, left=None, right=None):
        self.freq = freq
        self.char = char
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(s):
    freq = Counter(s)
    heap = [Node(f, c) for c, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.freq + right.freq, None, left, right)
        heapq.heappush(heap, merged)
    return heapq.heappop(heap) if heap else None
```

## Generate Codes (Traverse Tree)

DFS from root: left edge = 0, right edge = 1. At leaf, record code.

```python
def generate_codes(root, path="", codes=None):
    if codes is None:
        codes = {}
    if root.char is not None:
        codes[root.char] = path if path else "0"
        return codes
    generate_codes(root.left, path + "0", codes)
    generate_codes(root.right, path + "1", codes)
    return codes
```

## Encode String

Replace each character with its Huffman code.

```python
def encode(s, codes):
    return "".join(codes[c] for c in s)
```

## Decode String

Traverse tree: 0 go left, 1 go right; at leaf output character and reset to root.

```python
def decode(encoded, root):
    result = []
    node = root
    for bit in encoded:
        node = node.left if bit == "0" else node.right
        if node.char is not None:
            result.append(node.char)
            node = root
    return "".join(result)
```

## Optimality Proof (Prefix-Free Minimum Expected Length)

Huffman produces a prefix-free code with minimum expected length. Proof sketch: Any optimal tree can be transformed so the two least frequent symbols are siblings at deepest level. Huffman guarantees this by always merging the two smallest. Induction on alphabet size.

## Minimum Cost to Merge Stones (Generalized Huffman)

Merge k consecutive piles; cost = sum of merged piles. Min total cost to merge into one. When k=2, same as Huffman. For k>2, use similar greedy: merge k smallest.

```python
import heapq

def merge_stones(stones, k):
    if (len(stones) - 1) % (k - 1) != 0:
        return -1
    heap = list(stones)
    heapq.heapify(heap)
    total = 0
    while len(heap) >= k:
        s = 0
        for _ in range(k):
            s += heapq.heappop(heap)
        total += s
        heapq.heappush(heap, s)
    return total
```

## Connect Sticks with Minimum Cost

Connect sticks into one; cost = sum of lengths of two sticks merged. Min total cost. Same as Huffman: always merge two smallest.

```python
import heapq

def connect_sticks(sticks):
    heapq.heapify(sticks)
    total = 0
    while len(sticks) > 1:
        a = heapq.heappop(sticks)
        b = heapq.heappop(sticks)
        cost = a + b
        total += cost
        heapq.heappush(sticks, cost)
    return total
```

## Minimum Cost to Connect Ropes

Same as connect sticks: merge two smallest until one remains.

```python
import heapq

def min_cost_to_connect_ropes(ropes):
    heapq.heapify(ropes)
    total = 0
    while len(ropes) > 1:
        a = heapq.heappop(ropes)
        b = heapq.heappop(ropes)
        cost = a + b
        total += cost
        heapq.heappush(ropes, cost)
    return total
```
