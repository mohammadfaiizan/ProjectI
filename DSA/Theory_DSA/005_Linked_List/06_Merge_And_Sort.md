# Merge and Sort

## Theory: Merge Two Sorted Lists

Maintain two pointers, one for each list. Compare current elements and append the smaller to the result. When one list is exhausted, append the remainder of the other. Time O(n+m), space O(1) for iterative.

## Merge Two Sorted Lists (Iterative)

```python
def merge_sorted_iterative(l1, l2):
    dummy = Node(0)
    tail = dummy
    while l1 is not None and l2 is not None:
        if l1.data <= l2.data:
            tail.next = l1
            l1 = l1.next
        else:
            tail.next = l2
            l2 = l2.next
        tail = tail.next
    tail.next = l1 if l1 is not None else l2
    return dummy.next
```

## Merge Two Sorted Lists (Recursive)

```python
def merge_sorted_recursive(l1, l2):
    if l1 is None:
        return l2
    if l2 is None:
        return l1
    if l1.data <= l2.data:
        l1.next = merge_sorted_recursive(l1.next, l2)
        return l1
    else:
        l2.next = merge_sorted_recursive(l1, l2.next)
        return l2
```

## Merge k Sorted Lists (Min-Heap O(n log k))

Use a min-heap of size k. Initially push the first node of each list. Pop the minimum, append to result, push its next if exists. Each of n nodes is pushed and popped once. Heap operations are O(log k). Total O(n log k).

```python
import heapq

def merge_k_lists_heap(lists):
    heap = []
    for i, head in enumerate(lists):
        if head is not None:
            heapq.heappush(heap, (head.data, i, head))
    dummy = Node(0)
    tail = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        tail.next = node
        tail = tail.next
        if node.next is not None:
            heapq.heappush(heap, (node.next.data, i, node.next))
    return dummy.next
```

## Merge k Sorted Lists (Divide and Conquer)

Merge lists in pairs repeatedly until one list remains. Each level processes all n nodes. Log k levels. O(n log k).

```python
def merge_k_lists_divide_conquer(lists):
    if not lists:
        return None
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged.append(merge_sorted_iterative(l1, l2) if l2 else l1)
        lists = merged
    return lists[0]
```

## Merge Sort on Singly Linked List (Split at Mid, Merge)

Split the list at the middle using slow-fast pointers. Recursively sort both halves. Merge the sorted halves. O(n log n) time, O(log n) stack space.

```python
def find_middle(head):
    slow = head
    fast = head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow

def merge_sort_singly(head):
    if head is None or head.next is None:
        return head
    mid = find_middle(head)
    second = mid.next
    mid.next = None
    left = merge_sort_singly(head)
    right = merge_sort_singly(second)
    return merge_sorted_iterative(left, right)
```

## Merge Sort on Doubly Linked List

Same as singly: split, sort, merge. When merging doubly linked lists, also set prev pointers.

```python
class DNode:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

def merge_doubly(l1, l2):
    dummy = DNode(0)
    tail = dummy
    while l1 is not None and l2 is not None:
        if l1.data <= l2.data:
            tail.next = l1
            l1.prev = tail
            l1 = l1.next
        else:
            tail.next = l2
            l2.prev = tail
            l2 = l2.next
        tail = tail.next
    if l1 is not None:
        tail.next = l1
        l1.prev = tail
    else:
        tail.next = l2
        if l2 is not None:
            l2.prev = tail
    result = dummy.next
    if result is not None:
        result.prev = None
    return result

def merge_sort_doubly(head):
    if head is None or head.next is None:
        return head
    mid = find_middle_doubly(head)
    second = mid.next
    mid.next = None
    if second is not None:
        second.prev = None
    left = merge_sort_doubly(head)
    right = merge_sort_doubly(second)
    return merge_doubly(left, right)

def find_middle_doubly(head):
    slow = head
    fast = head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow
```

## Insertion Sort on Linked List

Build the result list by inserting each node in sorted position. For each node, find its position in the already-sorted prefix and insert. O(n^2) time, O(1) space.

```python
def insertion_sort_list(head):
    dummy = Node(0)
    curr = head
    while curr is not None:
        next_node = curr.next
        prev = dummy
        while prev.next is not None and prev.next.data < curr.data:
            prev = prev.next
        curr.next = prev.next
        prev.next = curr
        curr = next_node
    return dummy.next
```

## Sort List of 0s 1s 2s (Change Data)

Count 0s, 1s, 2s. Traverse and overwrite node data. O(n) time, O(1) space.

```python
def sort_012_data(head):
    count = [0, 0, 0]
    curr = head
    while curr is not None:
        count[curr.data] += 1
        curr = curr.next
    curr = head
    for i in range(3):
        for _ in range(count[i]):
            curr.data = i
            curr = curr.next
    return head
```

## Sort List of 0s 1s 2s (Change Links)

Create three dummy lists for 0, 1, 2. Traverse and append each node to the appropriate list. Concatenate the three lists. O(n) time, O(1) space.

```python
def sort_012_links(head):
    zero_dummy = Node(0)
    one_dummy = Node(0)
    two_dummy = Node(0)
    zero_tail = zero_dummy
    one_tail = one_dummy
    two_tail = two_dummy
    curr = head
    while curr is not None:
        next_node = curr.next
        curr.next = None
        if curr.data == 0:
            zero_tail.next = curr
            zero_tail = zero_tail.next
        elif curr.data == 1:
            one_tail.next = curr
            one_tail = one_tail.next
        else:
            two_tail.next = curr
            two_tail = two_tail.next
        curr = next_node
    zero_tail.next = one_dummy.next if one_dummy.next else two_dummy.next
    one_tail.next = two_dummy.next
    return zero_dummy.next
```

## Sort List Containing Two Sorted Halves

Find the split point (where order breaks). Merge the two sorted halves. O(n) time.

```python
def sort_two_sorted_halves(head):
    if head is None or head.next is None:
        return head
    curr = head
    while curr.next is not None and curr.data <= curr.next.data:
        curr = curr.next
    if curr.next is None:
        return head
    second = curr.next
    curr.next = None
    return merge_sorted_iterative(head, sort_two_sorted_halves(second))
```

## Partition List Around Value x (Stable)

Maintain two lists: one for elements < x, one for elements >= x. Preserve relative order. Concatenate. O(n) time, O(1) space.

```python
def partition_stable(head, x):
    less_dummy = Node(0)
    ge_dummy = Node(0)
    less_tail = less_dummy
    ge_tail = ge_dummy
    curr = head
    while curr is not None:
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

## Sort Absolute Sorted Linked List

List is sorted by absolute value. Negative numbers appear in reverse order. Separate negative and positive, reverse negatives, merge. O(n) time.

```python
def sort_absolute_sorted(head):
    neg_head = None
    pos_head = None
    pos_tail = None
    curr = head
    while curr is not None:
        next_node = curr.next
        curr.next = None
        if curr.data < 0:
            curr.next = neg_head
            neg_head = curr
        else:
            if pos_head is None:
                pos_head = pos_tail = curr
            else:
                pos_tail.next = curr
                pos_tail = curr
        curr = next_node
    neg_reversed = reverse_iterative(neg_head)
    return merge_sorted_iterative(neg_reversed, pos_head)
```
