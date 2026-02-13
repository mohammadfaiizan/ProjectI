# Two Pointers

## Theory: Slow-Fast Pointers

Use two pointers moving at different speeds. Slow moves 1 step, fast moves 2 steps per iteration. When fast reaches the end, slow is at the middle. Useful for finding middle, detecting cycles, and palindrome checks.

## Find Middle Node (Slow-Fast)

```python
def find_middle(head):
    if head is None:
        return None
    slow = head
    fast = head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow
```

## Find Middle in One Pass

Same as above - slow-fast gives middle in one pass. When fast reaches end (or one before), slow is at middle.

```python
def find_middle_one_pass(head):
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow
```

## Find Nth from End (Lead Pointer n Gap)

Place first pointer n steps ahead. Move both until first reaches end. Second points to nth from end.

```python
def find_nth_from_end(head, n):
    first = head
    for _ in range(n):
        if first is None:
            return None
        first = first.next
    second = head
    while first is not None:
        first = first.next
        second = second.next
    return second
```

## Remove Nth from End

Use dummy node. Place first pointer (n+1) steps ahead. Move both until first reaches end. Second's next is the node to remove. O(n) one pass.

```python
def remove_nth_from_end(head, n):
    dummy = Node(0)
    dummy.next = head
    first = dummy
    for _ in range(n + 1):
        first = first.next
    second = dummy
    while first is not None:
        first = first.next
        second = second.next
    second.next = second.next.next
    return dummy.next
```

## Detect Intersection Point (Two Pointer Equalization)

Find lengths of both lists. Advance the longer list by the difference. Then move both in step. They meet at intersection if it exists.

```python
def get_intersection_node(headA, headB):
    def length(h):
        n = 0
        while h:
            n += 1
            h = h.next
        return n

    lenA = length(headA)
    lenB = length(headB)
    currA = headA
    currB = headB
    if lenA > lenB:
        for _ in range(lenA - lenB):
            currA = currA.next
    else:
        for _ in range(lenB - lenA):
            currB = currB.next
    while currA and currB:
        if currA == currB:
            return currA
        currA = currA.next
        currB = currB.next
    return None
```

## Detect Intersection Using Length Difference

Same as above - equalize lengths then traverse in parallel.

```python
def intersection_length_diff(headA, headB):
    lenA = 0
    lenB = 0
    a = headA
    b = headB
    while a:
        lenA += 1
        a = a.next
    while b:
        lenB += 1
        b = b.next
    a = headA
    b = headB
    if lenA > lenB:
        for _ in range(lenA - lenB):
            a = a.next
    else:
        for _ in range(lenB - lenA):
            b = b.next
    while a and b:
        if a == b:
            return a
        a = a.next
        b = b.next
    return None
```

## Check Palindrome (Slow-Fast + Reverse)

Find middle with slow-fast. Reverse second half. Compare first half with reversed second half. Restore second half.

```python
def is_palindrome_two_pointer(head):
    if head is None or head.next is None:
        return True
    slow = head
    fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    second_head = slow.next
    slow.next = None
    rev_second = reverse_list(second_head)
    first = head
    second = rev_second
    result = True
    while second:
        if first.data != second.data:
            result = False
            break
        first = first.next
        second = second.next
    slow.next = reverse_list(rev_second)
    return result

def reverse_list(head):
    prev = None
    while head:
        next_node = head.next
        head.next = prev
        prev = head
        head = next_node
    return prev
```

## Reorder List (L0 Ln L1 Ln-1)

Find middle, reverse second half, interleave first and reversed second. L0, Ln, L1, Ln-1, ...

```python
def reorder_list(head):
    if head is None or head.next is None:
        return
    slow = head
    fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    second = slow.next
    slow.next = None
    second = reverse_list(second)
    first = head
    while second:
        temp1 = first.next
        temp2 = second.next
        first.next = second
        second.next = temp1
        first = temp1
        second = temp2
```

## Partition List Around x

Create two lists: elements < x and elements >= x. Concatenate.

```python
def partition_list(head, x):
    less_dummy = Node(0)
    ge_dummy = Node(0)
    less_tail = less_dummy
    ge_tail = ge_dummy
    curr = head
    while curr:
        next_node = curr.next
        curr.next = None
        if curr.data < x:
            less_tail.next = curr
            less_tail = less_tail.next
        else:
            ge_tail.next = curr
            ge_tail = ge_tail.next
        curr = next_node
    less_tail.next = ge_dummy.next
    return less_dummy.next
```

## Segregate Even and Odd Positioned Nodes

Place even-indexed nodes in one list, odd-indexed in another. Concatenate. Use two pointers to build both lists in one pass.

```python
def segregate_even_odd_position(head):
    if head is None or head.next is None:
        return head
    even_head = head
    odd_head = head.next
    even = even_head
    odd = odd_head
    while odd and odd.next:
        even.next = odd.next
        even = even.next
        odd.next = even.next
        odd = odd.next
    even.next = odd_head
    return even_head
```

## Find Starting Point of Loop

Floyd's cycle detection: after slow and fast meet, place one at head, move both one step. Meeting point is cycle start.

```python
def find_loop_start(head):
    slow = head
    fast = head
    while fast and fast.next:
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

## Split Circular Linked List into Two Halves

Use slow-fast to find mid. Set mid.next to None for first list. For second, traverse from mid.next to last and set last.next = mid.next (head of second). First half ends at mid, second starts at mid.next.

```python
def split_circular(head):
    if head is None or head.next == head:
        return head, None
    slow = head
    fast = head
    while fast.next != head and fast.next.next != head:
        slow = slow.next
        fast = fast.next.next
    if fast.next.next == head:
        fast = fast.next
    first_head = head
    second_head = slow.next
    slow.next = first_head
    fast.next = second_head
    return first_head, second_head
```

## Josephus Problem Using Circular List

N people in circle, eliminate every k-th person. Last remaining wins. Model as circular linked list, repeatedly remove k-th node until one remains.

```python
def josephus_circular(n, k):
    head = Node(1)
    curr = head
    for i in range(2, n + 1):
        curr.next = Node(i)
        curr = curr.next
    curr.next = head
    while curr.next != curr:
        for _ in range(k - 1):
            curr = curr.next
        curr.next = curr.next.next
    return curr.data
```
