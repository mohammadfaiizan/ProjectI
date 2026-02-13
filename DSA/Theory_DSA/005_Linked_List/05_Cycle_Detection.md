# Cycle Detection

## Theory: Floyd's Cycle Detection (Tortoise and Hare)

Floyd's algorithm uses two pointers: slow (moves 1 step) and fast (moves 2 steps). If there is a cycle, the fast pointer will eventually meet the slow pointer inside the cycle. If there is no cycle, fast will reach the end (None).

### Why It Works

Let the list have a non-cyclic part of length L and a cycle of length C. When slow enters the cycle (after L steps), fast is already inside. Let the distance of fast from slow when slow enters be D (0 <= D < C). Each step, this distance decreases by 1 (fast gains 1 on slow). So they meet after at most C steps. The meeting point is at distance (C - D) from the cycle start (when slow entered).

### Proof of Cycle Start

When slow and fast meet, let:
- x = distance from head to cycle start
- y = distance from cycle start to meeting point
- z = remaining cycle length (C - y)

When they meet: slow has traveled x + y, fast has traveled x + y + n*C for some n >= 1. Since fast = 2 * slow: 2(x + y) = x + y + n*C, so x + y = n*C, hence x = n*C - y = (n-1)*C + (C - y) = (n-1)*C + z. So a pointer from head and a pointer from meeting point, both moving 1 step, will meet at the cycle start after x steps.

## Detect if Cycle Exists

```python
def has_cycle(head):
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

## Find Start Node of Cycle (Mathematical Proof)

As proved above: after finding the meeting point, place one pointer at head and one at the meeting point. Move both one step at a time. They meet at the cycle start.

```python
def find_cycle_start(head):
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            ptr = head
            while ptr != slow:
                ptr = ptr.next
                slow = slow.next
            return ptr
    return None
```

## Find Length of Cycle

After detecting the cycle and finding the meeting point, traverse from the meeting point and count steps until returning to it.

```python
def find_cycle_length(head):
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            count = 1
            curr = slow.next
            while curr != slow:
                curr = curr.next
                count += 1
            return count
    return 0
```

## Remove the Cycle

Find the cycle start, then traverse from it to find the node whose next points to the start. Set that next to None.

```python
def remove_cycle(head):
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            ptr = head
            while ptr.next != slow.next:
                ptr = ptr.next
                slow = slow.next
            slow.next = None
            return
```

## Detect Cycle Using Hashing (Visited Set)

Traverse the list and add each node to a set. If we encounter a node already in the set, there is a cycle. O(n) time and O(n) space.

```python
def has_cycle_hashing(head):
    visited = set()
    curr = head
    while curr is not None:
        if curr in visited:
            return True
        visited.add(curr)
        curr = curr.next
    return False
```

## Happy Number (Cycle Detection on Number)

A number is happy if repeatedly replacing it with the sum of squares of its digits eventually reaches 1. If it enters a cycle (without 1), it is not happy. Use Floyd's cycle detection on the sequence of numbers.

```python
def is_happy(n):
    def next_val(num):
        total = 0
        while num > 0:
            total += (num % 10) ** 2
            num //= 10
        return total

    slow = n
    fast = n
    while True:
        slow = next_val(slow)
        fast = next_val(next_val(fast))
        if slow == 1:
            return True
        if slow == fast:
            return False
```

## Linked List Cycle II (Return Cycle Start)

```python
def detect_cycle_start(head):
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            ptr = head
            while ptr != slow:
                ptr = ptr.next
                slow = slow.next
            return ptr
    return None
```

## Circular Array Loop Detection

Given an array where each element indicates the next index (can be negative for backward), determine if there is a cycle. Cycle must have length > 1 and all elements in cycle must have same direction.

```python
def circular_array_loop(nums):
    n = len(nums)

    def next_index(i):
        return (i + nums[i]) % n

    def same_direction(i, j):
        return nums[i] * nums[j] > 0

    for i in range(n):
        if nums[i] == 0:
            continue
        slow = i
        fast = i
        while same_direction(i, next_index(fast)) and same_direction(i, next_index(next_index(fast))):
            slow = next_index(slow)
            fast = next_index(next_index(fast))
            if slow == fast:
                if slow != next_index(slow):
                    return True
                break
        curr = i
        val = nums[i]
        while nums[curr] * val > 0:
            nxt = next_index(curr)
            nums[curr] = 0
            curr = nxt
    return False
```
