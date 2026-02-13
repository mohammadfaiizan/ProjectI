# Basic Heap Operations

## Insert Element (Append + Sift Up)

Insert: append new element at end, then sift up to restore heap property.

```python
def sift_up(arr, i):
    while i > 0:
        parent = (i - 1) // 2
        if arr[i] >= arr[parent]:
            break
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent

def heap_insert_min(arr, val):
    arr.append(val)
    sift_up(arr, len(arr) - 1)
```

## Extract Min/Max (Swap Root with Last + Sift Down)

Extract: swap root with last element, pop last, then sift down from root.

```python
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

def heap_extract_min(arr):
    if not arr:
        return None
    arr[0], arr[-1] = arr[-1], arr[0]
    result = arr.pop()
    if arr:
        sift_down_min(arr, len(arr), 0)
    return result
```

## Peek

```python
def heap_peek(arr):
    return arr[0] if arr else None
```

## Sift Up Implementation (Min-Heap)

```python
def sift_up_min(arr, i):
    while i > 0:
        parent = (i - 1) // 2
        if arr[i] >= arr[parent]:
            break
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent
```

## Sift Up Implementation (Max-Heap)

```python
def sift_up_max(arr, i):
    while i > 0:
        parent = (i - 1) // 2
        if arr[i] <= arr[parent]:
            break
        arr[i], arr[parent] = arr[parent], arr[i]
        i = parent
```

## Sift Down Implementation (Min-Heap)

```python
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
```

## Sift Down Implementation (Max-Heap)

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
```

## Build Heap from Array (Bottom-Up O(n))

Start from last non-leaf node (n//2 - 1) and sift down each. Why O(n): at level h, there are at most n/2^(h+1) nodes, each sifts down at most h times. Sum = n * sum(h/2^h) = O(n).

```python
def build_min_heap(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift_down_min(arr, n, i)
```

## Min-Heap Class from Scratch

```python
class MinHeap:
    def __init__(self):
        self.arr = []

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.arr[i] >= self.arr[parent]:
                break
            self.arr[i], self.arr[parent] = self.arr[parent], self.arr[i]
            i = parent

    def _sift_down(self, i):
        n = len(self.arr)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self.arr[left] < self.arr[smallest]:
                smallest = left
            if right < n and self.arr[right] < self.arr[smallest]:
                smallest = right
            if smallest == i:
                break
            self.arr[i], self.arr[smallest] = self.arr[smallest], self.arr[i]
            i = smallest

    def push(self, val):
        self.arr.append(val)
        self._sift_up(len(self.arr) - 1)

    def pop(self):
        if not self.arr:
            return None
        self.arr[0], self.arr[-1] = self.arr[-1], self.arr[0]
        result = self.arr.pop()
        if self.arr:
            self._sift_down(0)
        return result

    def peek(self):
        return self.arr[0] if self.arr else None

    def size(self):
        return len(self.arr)

    def is_empty(self):
        return len(self.arr) == 0
```

## Max-Heap Class from Scratch

```python
class MaxHeap:
    def __init__(self):
        self.arr = []

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.arr[i] <= self.arr[parent]:
                break
            self.arr[i], self.arr[parent] = self.arr[parent], self.arr[i]
            i = parent

    def _sift_down(self, i):
        n = len(self.arr)
        while True:
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self.arr[left] > self.arr[largest]:
                largest = left
            if right < n and self.arr[right] > self.arr[largest]:
                largest = right
            if largest == i:
                break
            self.arr[i], self.arr[largest] = self.arr[largest], self.arr[i]
            i = largest

    def push(self, val):
        self.arr.append(val)
        self._sift_up(len(self.arr) - 1)

    def pop(self):
        if not self.arr:
            return None
        self.arr[0], self.arr[-1] = self.arr[-1], self.arr[0]
        result = self.arr.pop()
        if self.arr:
            self._sift_down(0)
        return result

    def peek(self):
        return self.arr[0] if self.arr else None

    def size(self):
        return len(self.arr)

    def is_empty(self):
        return len(self.arr) == 0
```

## Convert Min-Heap to Max-Heap

Rebuild from scratch by sifting down from last non-leaf in max-heap order.

```python
def min_heap_to_max_heap(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        sift_down_max(arr, n, i)
```

## Check if Array is Valid Heap

```python
def is_valid_min_heap(arr):
    n = len(arr)
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] < arr[i]:
            return False
        if right < n and arr[right] < arr[i]:
            return False
    return True

def is_valid_max_heap(arr):
    n = len(arr)
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and arr[left] > arr[i]:
            return False
        if right < n and arr[right] > arr[i]:
            return False
    return True
```

## Find Minimum in Max-Heap

Minimum must be in a leaf. Leaves are indices from n//2 to n-1. Scan leaves: O(n/2) = O(n).

```python
def find_min_in_max_heap(arr):
    if not arr:
        return None
    n = len(arr)
    min_val = arr[n // 2]
    for i in range(n // 2, n):
        min_val = min(min_val, arr[i])
    return min_val
```

## Find Maximum in Min-Heap

Maximum must be in a leaf. Same approach.

```python
def find_max_in_min_heap(arr):
    if not arr:
        return None
    n = len(arr)
    max_val = arr[n // 2]
    for i in range(n // 2, n):
        max_val = max(max_val, arr[i])
    return max_val
```

## Get Size and isEmpty

```python
def heap_size(arr):
    return len(arr)

def heap_is_empty(arr):
    return len(arr) == 0
```
