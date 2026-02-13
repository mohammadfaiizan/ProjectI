# Reversal Techniques

## Theory: Reverse Entire Singly (Iterative prev/curr/next)

Maintain three pointers: `prev` (initially None), `curr` (current node), and `next_node` (to save the next link before breaking it). In each step: save next, reverse the link (curr.next = prev), advance prev and curr. When curr becomes None, prev is the new head.

## Reverse Entire Singly (Iterative)

```python
def reverse_iterative(head):
    prev = None
    curr = head
    while curr is not None:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

## Reverse Entire Singly (Recursive)

```python
def reverse_recursive(head):
    if head is None or head.next is None:
        return head
    rest = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return rest
```

## Reverse Doubly Linked List

```python
class DNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

def reverse_doubly(head):
    curr = head
    while curr is not None:
        curr.prev, curr.next = curr.next, curr.prev
        head = curr
        curr = curr.prev
    return head
```

## Reverse in Groups of k (Iterative)

```python
def reverse_k_group_iterative(head, k):
    dummy = Node(0)
    dummy.next = head
    group_prev = dummy
    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if kth is None:
                return dummy.next
        group_next = kth.next
        prev = group_next
        curr = group_prev.next
        while curr != group_next:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        temp = group_prev.next
        group_prev.next = kth
        group_prev = temp
    return dummy.next
```

## Reverse in Groups of k (Recursive)

```python
def reverse_k_group_recursive(head, k):
    curr = head
    count = 0
    while curr is not None and count < k:
        curr = curr.next
        count += 1
    if count == k:
        prev = reverse_k_group_recursive(curr, k)
        curr = head
        for _ in range(k):
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        head = prev
    return head
```

## Reverse Between Positions m and n (One-Pass)

```python
def reverse_between_one_pass(head, m, n):
    if head is None or m == n:
        return head
    dummy = Node(0)
    dummy.next = head
    prev = dummy
    for _ in range(m - 1):
        prev = prev.next
    curr = prev.next
    for _ in range(n - m):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node
    return dummy.next
```

## Reverse Alternating k Nodes

```python
def reverse_alternating_k(head, k):
    if head is None:
        return None
    curr = head
    prev = None
    for _ in range(k):
        if curr is None:
            break
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    head.next = curr
    for _ in range(k - 1):
        if curr is None:
            break
        curr = curr.next
    if curr is not None:
        curr.next = reverse_alternating_k(curr.next, k)
    return prev
```

## Reverse First k Nodes

```python
def reverse_first_k(head, k):
    if head is None or k <= 1:
        return head
    curr = head
    prev = None
    count = 0
    while curr is not None and count < k:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
        count += 1
    head.next = curr
    return prev
```

## Reverse Last k Nodes

```python
def reverse_last_k(head, k):
    if head is None or k <= 0:
        return head
    n = 0
    curr = head
    while curr is not None:
        n += 1
        curr = curr.next
    if k >= n:
        return reverse_iterative(head)
    skip = n - k
    curr = head
    for _ in range(skip - 1):
        curr = curr.next
    new_head = reverse_iterative(curr.next)
    curr.next = new_head
    return head
```

## Check Palindrome via Reverse-and-Compare

```python
def is_palindrome_reverse_compare(head):
    if head is None or head.next is None:
        return True
    slow = head
    fast = head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    second_half = reverse_iterative(slow.next)
    first = head
    second = second_half
    result = True
    while second is not None:
        if first.data != second.data:
            result = False
            break
        first = first.next
        second = second.next
    reverse_iterative(second_half)
    return result
```

## Pairwise Swap (Reverse Pairs)

```python
def pairwise_swap(head):
    if head is None or head.next is None:
        return head
    dummy = Node(0)
    dummy.next = head
    prev = dummy
    while prev.next is not None and prev.next.next is not None:
        first = prev.next
        second = first.next
        prev.next = second
        first.next = second.next
        second.next = first
        prev = first
    return dummy.next
```

## Reverse Using Stack

```python
def reverse_using_stack(head):
    if head is None:
        return None
    stack = []
    curr = head
    while curr is not None:
        stack.append(curr)
        curr = curr.next
    head = stack.pop()
    curr = head
    while stack:
        curr.next = stack.pop()
        curr = curr.next
    curr.next = None
    return head
```

## Rearrange List (First Last Second Second-Last)

```python
def rearrange_first_last(head):
    if head is None or head.next is None:
        return head
    slow = head
    fast = head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    second = reverse_iterative(slow.next)
    slow.next = None
    first = head
    dummy = Node(0)
    tail = dummy
    while first is not None or second is not None:
        if first is not None:
            tail.next = first
            tail = tail.next
            first = first.next
        if second is not None:
            tail.next = second
            tail = tail.next
            second = second.next
    return dummy.next
```

## Fold Linked List (Interleave First Half with Reversed Second Half)

```python
def fold_list(head):
    if head is None or head.next is None:
        return head
    slow = head
    fast = head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    second = reverse_iterative(slow.next)
    slow.next = None
    first = head
    while second is not None:
        temp1 = first.next
        temp2 = second.next
        first.next = second
        second.next = temp1
        first = temp1
        second = temp2
    return head
```
