# Advanced Heap Operations

## Delete Arbitrary Element by Index

Replace element with last, pop last, then sift up or sift down depending on new value vs old.

```python
def sift_up_min(arr, i):
    while i > 0:
        parent = (i - 1) // 2
        if arr[i] >= arr[parent]:
            break
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent

def sift_down_min(arr, n, i):
    while True:
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] < arr[smallest]:
            smallest = left
        if right < n and arr[right] < arr[smallest]:
            smallest = right
        if smallest == i:
            break
        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest

def heap_delete_by_index(arr, idx):
    if idx < 0 or idx >= len(arr):
        return False
    last = arr.pop()
    if idx == len(arr):
        return True
    arr[idx] = last
    n = len(arr)
    if idx > 0 and arr[idx] < arr[(idx - 1) // 2]:
        sift_up_min(arr, idx)
    else:
        sift_down_min(arr, n, idx)
    return True
```

## Delete Arbitrary Element by Value

Linear search for value, then delete by index. For repeated values, delete first occurrence.

```python
def heap_delete_by_value(arr, val):
    try:
        idx = arr.index(val)
        return heap_delete_by_index(arr, idx)
    except ValueError:
        return False
```

## Decrease Key (Min-Heap)

Decreasing a key may violate heap property with parent. Sift up.

```python
def decrease_key_min(arr, idx, new_val):
    if idx < 0 or idx >= len(arr) or new_val >= arr[idx]:
        return False
    arr[idx] = new_val
    sift_up_min(arr, idx)
    return True
```

## Increase Key (Max-Heap)

Increasing a key may violate heap property with parent. Sift up in max-heap.

```python
def sift_up_max(arr, i):
    while i > 0:
        parent = (i - 1) // 2
        if arr[i] <= arr[parent]:
            break
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent

def increase_key_max(arr, idx, new_val):
    if idx < 0 or idx >= len(arr) or new_val <= arr[idx]:
        return False
    arr[idx] = new_val
    sift_up_max(arr, idx)
    return True
```

## Merge Two Heaps

Concatenate arrays and rebuild heap from scratch. O(n + m).

```python
def merge_min_heaps(arr1, arr2):
    merged = arr1 + arr2
    n = len(merged)
    for i in range(n // 2 - 1, -1, -1):
        sift_down_min(merged, n, i)
    return merged
```

## Heap Sort (Build Max-Heap + Extract)

```python
def sift_down_max(arr, n, i):
    while True:
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[largest]:
            largest = left
        if right < n and arr[right] > arr[largest]:
            largest = right
        if largest == i:
            break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift_down_max(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        sift_down_max(arr, i, 0)
```

## K-Way Merge Using Heap

Merge k sorted arrays. Use min-heap of size k with (value, array_index, element_index).

```python
import heapq

def k_way_merge(arrays):
    heap = []
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))
    result = []
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))
    return result
```

## Replace Root (Extract + Insert Optimized)

Replace root with new value and sift down once. O(log n) vs two O(log n) operations.

```python
def replace_root_min(arr, new_val):
    if not arr:
        arr.append(new_val)
        return None
    old = arr[0]
    arr[0] = new_val
    sift_down_min(arr, len(arr), 0)
    return old
```

## Priority Queue with Custom Comparator

Use a wrapper that inverts comparison for max-heap behavior, or pass a key function.

```python
import heapq

class PriorityQueue:
    def __init__(self, key=lambda x: x):
        self.heap = []
        self.key = key

    def push(self, item):
        heapq.heappush(self.heap, (self.key(item), item))

    def pop(self):
        if not self.heap:
            return None
        _, item = heapq.heappop(self.heap)
        return item

    def peek(self):
        return self.heap[0][1] if self.heap else None

    def is_empty(self):
        return len(self.heap) == 0
```

## Indexed Priority Queue

Maintains a mapping from element/key to heap index for O(log n) update and delete.

```python
class IndexedMinHeap:
    def __init__(self):
        self.values = []
        self.index_map = {}

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.values[i][0] >= self.values[parent][0]:
                break
            ki, kp = self.values[i][1], self.values[parent][1]
            self.values[i], self.values[parent] = self.values[parent], self.values[i]
            self.index_map[ki], self.index_map[kp] = parent, i
            i = parent

    def _sift_down(self, i):
        n = len(self.values)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self.values[left][0] < self.values[smallest][0]:
                smallest = left
            if right < n and self.values[right][0] < self.values[smallest][0]:
                smallest = right
            if smallest == i:
                break
            ki, ks = self.values[i][1], self.values[smallest][1]
            self.values[i], self.values[smallest] = self.values[smallest], self.values[i]
            self.index_map[ki], self.index_map[ks] = smallest, i
            i = smallest

    def push(self, key, value):
        if key in self.index_map:
            self.update(key, value)
            return
        i = len(self.values)
        self.values.append((value, key))
        self.index_map[key] = i
        self._sift_up(i)

    def pop(self):
        if not self.values:
            return None
        val, key = self.values[0]
        del self.index_map[key]
        last = self.values.pop()
        if self.values:
            self.values[0] = last
            self.index_map[last[1]] = 0
            self._sift_down(0)
        return key, val

    def update(self, key, new_value):
        if key not in self.index_map:
            self.push(key, new_value)
            return
        i = self.index_map[key]
        old_val = self.values[i][0]
        self.values[i] = (new_value, key)
        if new_value < old_val:
            self._sift_up(i)
        else:
            self._sift_down(i)

    def get(self, key):
        i = self.index_map.get(key)
        return self.values[i][0] if i is not None else None

    def contains(self, key):
        return key in self.index_map
```

## D-Ary Heap

Each node has d children. Parent at i has children at d*i+1, d*i+2, ..., d*i+d. Parent of i is (i-1)//d.

```python
def d_ary_parent(i, d):
    return (i - 1) // d

def d_ary_child(i, k, d):
    return d * i + k

def d_ary_sift_down_min(arr, n, i, d):
    while True:
        smallest = i
        for k in range(1, d + 1):
            child = d * i + k
            if child < n and arr[child] < arr[smallest]:
                smallest = child
        if smallest == i:
            break
        arr[i], arr[smallest] = arr[smallest], arr[i]
        i = smallest

def d_ary_sift_up_min(arr, i, d):
    while i > 0:
        parent = (i - 1) // d
        if arr[i] >= arr[parent]:
            break
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent

class DaryMinHeap:
    def __init__(self, d=2):
        self.arr = []
        self.d = d

    def push(self, val):
        self.arr.append(val)
        d_ary_sift_up_min(self.arr, len(self.arr) - 1, self.d)

    def pop(self):
        if not self.arr:
            return None
        self.arr[0], self.arr[-1] = self.arr[-1], self.arr[0]
        result = self.arr.pop()
        if self.arr:
            d_ary_sift_down_min(self.arr, len(self.arr), 0, self.d)
        return result
```

## Fibonacci Heap Overview

- Amortized O(1) insert, decrease key, merge
- Amortized O(log n) extract min, delete
- Structure: collection of heap-ordered trees with lazy consolidation
- Used in Dijkstra and Prim for theoretical speedup

## Binomial Heap Overview

- Union in O(log n)
- Insert O(1) amortized
- Extract min O(log n)
- Structure: forest of binomial trees B0, B1, B2, ... (Bk has 2^k nodes)

## Median Maintenance (Two Heaps)

```python
import heapq

class MedianMaintainer:
    def __init__(self):
        self.lo = []
        self.hi = []

    def add(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def get_median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

## Lazy Deletion in Heap

Mark elements as deleted instead of removing. On pop, skip deleted elements.

```python
import heapq

class LazyHeap:
    def __init__(self):
        self.heap = []
        self.deleted = set()
        self.counter = 0

    def push(self, val):
        heapq.heappush(self.heap, (val, self.counter))
        self.counter += 1

    def delete(self, val):
        self.deleted.add(val)

    def pop(self):
        while self.heap:
            val, _ = heapq.heappop(self.heap)
            if val not in self.deleted:
                self.deleted.discard(val)
                return val
        return None

    def peek(self):
        while self.heap and self.heap[0][0] in self.deleted:
            heapq.heappop(self.heap)
        return self.heap[0][0] if self.heap else None
```
