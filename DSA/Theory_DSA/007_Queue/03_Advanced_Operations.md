# Queue Advanced Operations

## Circular Deque

A double-ended queue implemented as a circular buffer. Supports O(1) operations at both ends.

```python
class CircularDeque:
    def __init__(self, capacity):
        self.capacity = capacity
        self.arr = [None] * capacity
        self.front = 0
        self.rear = 0
        self.size = 0

    def _prev(self, i):
        return (i - 1 + self.capacity) % self.capacity

    def _next(self, i):
        return (i + 1) % self.capacity

    def insertFront(self, value):
        if self.isFull():
            return False
        self.front = self._prev(self.front)
        self.arr[self.front] = value
        self.size += 1
        return True

    def insertLast(self, value):
        if self.isFull():
            return False
        self.arr[self.rear] = value
        self.rear = self._next(self.rear)
        self.size += 1
        return True

    def deleteFront(self):
        if self.isEmpty():
            return False
        self.front = self._next(self.front)
        self.size -= 1
        return True

    def deleteLast(self):
        if self.isEmpty():
            return False
        self.rear = self._prev(self.rear)
        self.size -= 1
        return True

    def getFront(self):
        if self.isEmpty():
            return -1
        return self.arr[self.front]

    def getRear(self):
        if self.isEmpty():
            return -1
        return self.arr[self._prev(self.rear)]

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity
```

## Design Hit Counter

Count hits in the last 300 seconds. Use a queue to store timestamps; on getHits, remove timestamps older than 300 seconds, then return queue size.

```python
from collections import deque

class HitCounter:
    def __init__(self):
        self.q = deque()

    def hit(self, timestamp):
        self.q.append(timestamp)

    def getHits(self, timestamp):
        while self.q and self.q[0] <= timestamp - 300:
            self.q.popleft()
        return len(self.q)
```

## Design Recent Calls Counter

Count calls in the last 3000 ms. Same pattern as hit counter.

```python
from collections import deque

class RecentCounter:
    def __init__(self):
        self.q = deque()

    def ping(self, t):
        self.q.append(t)
        while self.q[0] < t - 3000:
            self.q.popleft()
        return len(self.q)
```

## Moving Average from Data Stream

Maintain a sliding window of size `size`. Use a queue to store elements; when full, dequeue oldest before enqueueing new. Return sum/window_size.

```python
from collections import deque

class MovingAverage:
    def __init__(self, size):
        self.size = size
        self.q = deque()
        self.total = 0

    def next(self, val):
        if len(self.q) == self.size:
            self.total -= self.q.popleft()
        self.q.append(val)
        self.total += val
        return self.total / len(self.q)
```

## LRU Cache Using Queue and HashMap

Least Recently Used cache: when capacity exceeded, evict least recently used. Use a queue (deque) for access order and a hashmap for O(1) lookup. On get: move to end (recent). On put: if key exists, move to end; else add and evict from front if full.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

Using deque and dict (manual move to end):

```python
from collections import deque

class LRUCacheDeque:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = deque()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.popleft()
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
```

## Implement Stack Using Queues

Push O(n): enqueue new, then rotate n-1 elements. Pop O(1): dequeue.

```python
from collections import deque

class StackUsingQueues:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        if not self.q:
            raise IndexError("pop from empty stack")
        return self.q.popleft()

    def top(self):
        if not self.q:
            raise IndexError("top from empty stack")
        return self.q[0]

    def empty(self):
        return len(self.q) == 0
```

## Interleave First and Second Half of Queue

Given queue [1,2,3,4,5,6], produce [1,4,2,5,3,6]. Push first half to stack; enqueue stack to queue; rotate second half to front; push new first half to stack; interleave by alternating pop and dequeue.

```python
from collections import deque

def interleave_queue(q):
    n = len(q)
    half = n // 2
    stack = []
    for _ in range(half):
        stack.append(q.popleft())
    while stack:
        q.append(stack.pop())
    for _ in range(n - half):
        q.append(q.popleft())
    for _ in range(half):
        stack.append(q.popleft())
    for _ in range(half):
        q.append(stack.pop())
        q.append(q.popleft())
```

## Reverse First K Elements of Queue

Reverse the order of the first k elements. Use a stack: push first k, pop back to queue, then rotate remaining n-k to the back.

```python
from collections import deque

def reverse_first_k(q, k):
    if k <= 0 or k > len(q):
        return
    stack = []
    for _ in range(k):
        stack.append(q.popleft())
    while stack:
        q.append(stack.pop())
    for _ in range(len(q) - k):
        q.append(q.popleft())
```

## Sort a Queue

Sort queue using only queue operations and one extra queue. Repeatedly find minimum by rotating, remove it, append to result queue, then copy result back.

```python
from collections import deque

def sort_queue(q):
    result = deque()
    n = len(q)
    for _ in range(n):
        min_val = float('inf')
        for _ in range(len(q)):
            val = q.popleft()
            min_val = min(min_val, val)
            q.append(val)
        removed = False
        for _ in range(len(q)):
            val = q.popleft()
            if val == min_val and not removed:
                removed = True
            else:
                q.append(val)
        result.append(min_val)
    while result:
        q.append(result.popleft())
```

## Generate Binary Numbers 1 to N Using Queue

Generate binary numbers from "1" to n by appending "0" and "1" to each number and enqueueing.

```python
from collections import deque

def generate_binary_numbers(n):
    result = []
    q = deque(["1"])
    for _ in range(n):
        curr = q.popleft()
        result.append(curr)
        q.append(curr + "0")
        q.append(curr + "1")
    return result
```

## First Non-Repeating Character in Stream

For each character seen, return first non-repeating so far. Use a queue for candidates and a count/freq map. When we see a char, increment count; dequeue while front has count > 1.

```python
from collections import deque

def first_non_repeating_stream(stream):
    freq = {}
    q = deque()
    result = []
    for c in stream:
        freq[c] = freq.get(c, 0) + 1
        q.append(c)
        while q and freq[q[0]] > 1:
            q.popleft()
        result.append(q[0] if q else -1)
    return result
```

## Design Task Scheduler

Schedule tasks with cooldown n. Same task cannot run within n slots. Use a max-heap for task counts and a queue for (task, available_time). Each slot: if queue front available, add back to heap; pop from heap, schedule, push to queue with next_available = time + n + 1.

```python
import heapq
from collections import deque, Counter

def least_interval(tasks, n):
    counts = Counter(tasks)
    heap = [-c for c in counts.values()]
    heapq.heapify(heap)
    q = deque()
    time = 0
    while heap or q:
        time += 1
        if heap:
            cnt = heapq.heappop(heap) + 1
            if cnt != 0:
                q.append((cnt, time + n))
        if q and q[0][1] == time:
            heapq.heappush(heap, q.popleft()[0])
    return time
```

## Sliding Window Maximum Using Deque

For each window of size k, return the maximum. Maintain a deque of indices in decreasing order of values. When sliding: remove indices outside window from front; remove indices with smaller values from back; add current index; front is max.

```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()
    result = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(nums[dq[0]])
    return result
```

## Implement Priority Queue (Simple Sorted List)

Simple implementation using a sorted list. Insert O(n), extract_min O(1) or O(n) if we pop from front.

```python
class SimplePriorityQueue:
    def __init__(self):
        self.data = []

    def push(self, x):
        self.data.append(x)
        self.data.sort()

    def pop(self):
        if not self.data:
            raise IndexError("pop from empty priority queue")
        return self.data.pop(0)

    def peek(self):
        if not self.data:
            raise IndexError("peek from empty priority queue")
        return self.data[0]

    def empty(self):
        return len(self.data) == 0
```

## First Unique Character in Data Stream

Similar to first non-repeating. Maintain queue of unique candidates and a seen/freq map.

```python
from collections import deque

class FirstUnique:
    def __init__(self, nums):
        self.q = deque()
        self.freq = {}
        for x in nums:
            self.add(x)

    def showFirstUnique(self):
        while self.q and self.freq[self.q[0]] > 1:
            self.q.popleft()
        return self.q[0] if self.q else -1

    def add(self, value):
        self.freq[value] = self.freq.get(value, 0) + 1
        self.q.append(value)
```

## Maximum of All Subarrays of Size K (Deque)

Same as sliding window maximum.

```python
from collections import deque

def max_of_subarrays(arr, k):
    dq = deque()
    result = []
    for i, x in enumerate(arr):
        while dq and arr[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result
```

## Time Needed to Buy Tickets

People in queue, each needs tickets[i] tickets. Person at front buys 1 per second. Return seconds for person at position k to finish. Simulate: each second, front person buys 1; if they are done, they leave; else they go to back. Count seconds until person k finishes.

```python
from collections import deque

def time_required_to_buy(tickets, k):
    q = deque((i, t) for i, t in enumerate(tickets))
    time = 0
    while True:
        i, t = q.popleft()
        time += 1
        t -= 1
        if t > 0:
            q.append((i, t))
        elif i == k:
            return time
```
