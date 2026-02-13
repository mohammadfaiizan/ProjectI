# Advanced Hashing Operations

## LRU Cache (Hash Map + Doubly Linked List)

LRU evicts least recently used item when capacity is exceeded. Hash map gives O(1) lookup; doubly linked list gives O(1) move to front and removal from tail.

```python
class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add(self, node):
        p = self.tail.prev
        p.next = node
        node.prev = p
        node.next = self.tail
        self.tail.prev = node

    def _remove(self, node):
        p, n = node.prev, node.next
        p.next = n
        n.prev = p

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add(node)
        return node.val

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add(node)
        self.cache[key] = node
        if len(self.cache) > self.cap:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]
```

## LFU Cache

LFU evicts least frequently used. On tie, evict LRU. Uses: freq -> doubly linked list of nodes, key -> node, min_freq tracker.

```python
class LFUNode:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.freq = 1
        self.prev = None
        self.next = None

class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.key_to_node = {}
        self.freq_to_dll = {}
        self.min_freq = 0

    def _add_to_dll(self, freq, node):
        if freq not in self.freq_to_dll:
            head, tail = LFUNode(0, 0), LFUNode(0, 0)
            head.next = tail
            tail.prev = head
            self.freq_to_dll[freq] = (head, tail)
        head, tail = self.freq_to_dll[freq]
        p = tail.prev
        p.next = node
        node.prev = p
        node.next = tail
        tail.prev = node

    def _remove_from_dll(self, node):
        p, n = node.prev, node.next
        p.next = n
        n.prev = p

    def get(self, key):
        if key not in self.key_to_node:
            return -1
        node = self.key_to_node[key]
        self._remove_from_dll(node)
        node.freq += 1
        self._add_to_dll(node.freq, node)
        if self.min_freq == node.freq - 1:
            head, tail = self.freq_to_dll[node.freq - 1]
            if head.next == tail:
                self.min_freq = node.freq
        return node.val

    def put(self, key, value):
        if self.cap == 0:
            return
        if key in self.key_to_node:
            node = self.key_to_node[key]
            node.val = value
            self.get(key)
            return
        if len(self.key_to_node) >= self.cap:
            head, tail = self.freq_to_dll[self.min_freq]
            lfu = head.next
            self._remove_from_dll(lfu)
            del self.key_to_node[lfu.key]
        node = LFUNode(key, value)
        self.min_freq = 1
        self._add_to_dll(1, node)
        self.key_to_node[key] = node
```

## Consistent Hashing Overview

Distributes keys across nodes in a ring. When a node is added/removed, only keys near that node are remapped. Used in distributed caches (e.g., Redis cluster).

Concept: hash keys and nodes to positions on a circle. Key goes to first node clockwise from its position.

## Bloom Filter Overview

Probabilistic structure for set membership. No false negatives; possible false positives. Uses k hash functions and m bits. Space-efficient for large sets.

```python
class BloomFilter:
    def __init__(self, size, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = [False] * size

    def _hashes(self, key):
        h1 = hash(key) % self.size
        for i in range(self.num_hashes):
            yield (h1 + i * (hash(str(key) + str(i)) % self.size)) % self.size

    def add(self, key):
        for h in self._hashes(key):
            self.bits[h] = True

    def might_contain(self, key):
        return all(self.bits[h] for h in self._hashes(key))
```

## Count-Min Sketch Overview

Probabilistic structure for frequency estimation. Overcounts but never undercounts. Uses d rows of w counters with different hash functions.

## Design HashMap Class

```python
class MyHashMap:
    def __init__(self):
        self.cap = 1000
        self.buckets = [[] for _ in range(self.cap)]

    def _hash(self, key):
        return key % self.cap

    def put(self, key, value):
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)
                return
        self.buckets[idx].append((key, value))

    def get(self, key):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return -1

    def remove(self, key):
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx].pop(i)
                return
```

## Design HashSet Class

```python
class MyHashSet:
    def __init__(self):
        self.cap = 1000
        self.buckets = [set() for _ in range(self.cap)]

    def _hash(self, key):
        return key % self.cap

    def add(self, key):
        self.buckets[self._hash(key)].add(key)

    def remove(self, key):
        self.buckets[self._hash(key)].discard(key)

    def contains(self, key):
        return key in self.buckets[self._hash(key)]
```

## Time-Based Key-Value Store

Store multiple versions per key. get(key, timestamp) returns value with largest timestamp <= timestamp.

```python
from collections import defaultdict
import bisect

class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)

    def set(self, key, value, timestamp):
        self.store[key].append((timestamp, value))

    def get(self, key, timestamp):
        arr = self.store[key]
        i = bisect.bisect_right(arr, (timestamp, chr(127)))
        if i == 0:
            return ""
        return arr[i - 1][1]
```

## Design Underground System

Track check-in station/time and compute average time between two stations.

```python
from collections import defaultdict

class UndergroundSystem:
    def __init__(self):
        self.check_in = {}
        self.trips = defaultdict(lambda: [0, 0])

    def checkIn(self, id, stationName, t):
        self.check_in[id] = (stationName, t)

    def checkOut(self, id, stationName, t):
        start, t0 = self.check_in.pop(id)
        key = (start, stationName)
        self.trips[key][0] += t - t0
        self.trips[key][1] += 1

    def getAverageTime(self, startStation, endStation):
        total, count = self.trips[(startStation, endStation)]
        return total / count
```

## Snapshot Array

Support set, get, and snapshot. get(index, snap_id) returns value at index when snapshot snap_id was taken.

```python
from bisect import bisect_right

class SnapshotArray:
    def __init__(self, length):
        self.arr = [[(0, 0)] for _ in range(length)]
        self.snap_id = 0

    def set(self, index, val):
        self.arr[index].append((self.snap_id, val))

    def snap(self):
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index, snap_id):
        hist = self.arr[index]
        i = bisect_right(hist, (snap_id, 10**9))
        return hist[i - 1][1]
```

## Encode and Decode TinyURL

```python
import random
import string

class Codec:
    def __init__(self):
        self.long_to_short = {}
        self.short_to_long = {}
        self.chars = string.ascii_letters + string.digits

    def _gen(self):
        return ''.join(random.choice(self.chars) for _ in range(6))

    def encode(self, longUrl):
        if longUrl in self.long_to_short:
            return self.long_to_short[longUrl]
        short = self._gen()
        while short in self.short_to_long:
            short = self._gen()
        self.long_to_short[longUrl] = short
        self.short_to_long[short] = longUrl
        return short

    def decode(self, shortUrl):
        return self.short_to_long.get(shortUrl, "")
```

## Insert Delete GetRandom O(1)

Use list for random index, dict for O(1) lookup. On delete, swap with last and pop.

```python
import random

class RandomizedSet:
    def __init__(self):
        self.d = {}
        self.arr = []

    def insert(self, val):
        if val in self.d:
            return False
        self.d[val] = len(self.arr)
        self.arr.append(val)
        return True

    def remove(self, val):
        if val not in self.d:
            return False
        i = self.d[val]
        last = self.arr[-1]
        self.arr[i] = last
        self.d[last] = i
        self.arr.pop()
        del self.d[val]
        return True

    def getRandom(self):
        return random.choice(self.arr)
```

## Insert Delete GetRandom O(1) with Duplicates

Store indices in a set per value to support duplicates.

```python
import random

class RandomizedCollection:
    def __init__(self):
        self.d = {}
        self.arr = []

    def insert(self, val):
        if val not in self.d:
            self.d[val] = set()
        self.d[val].add(len(self.arr))
        self.arr.append(val)
        return len(self.d[val]) == 1

    def remove(self, val):
        if val not in self.d or not self.d[val]:
            return False
        i = self.d[val].pop()
        last = self.arr[-1]
        self.arr[i] = last
        self.d[last].add(i)
        self.d[last].discard(len(self.arr) - 1)
        self.arr.pop()
        if not self.d[val]:
            del self.d[val]
        return True

    def getRandom(self):
        return random.choice(self.arr)
```
