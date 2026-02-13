# Advanced Linked List Operations

## Reverse Entire List (Iterative Three-Pointer)

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

## Reverse Entire List (Recursive)

```python
def reverse_recursive(head):
    if head is None or head.next is None:
        return head
    rest = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return rest
```

## Reverse in Groups of k

```python
def reverse_k_group(head, k):
    curr = head
    count = 0
    while curr is not None and count < k:
        curr = curr.next
        count += 1
    if count == k:
        prev = reverse_k_group(curr, k)
        curr = head
        for _ in range(k):
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        head = prev
    return head
```

## Reverse Between Positions m and n

```python
def reverse_between(head, m, n):
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

## Reverse Alternating k-Group

```python
def reverse_alternating_k_group(head, k):
    curr = head
    prev = None
    while curr is not None:
        group_head = curr
        count = 0
        while curr is not None and count < k:
            curr = curr.next
            count += 1
        if count == k:
            rev_head = reverse_sublist(group_head, k)
            if prev is None:
                head = rev_head
            else:
                prev.next = rev_head
            group_head.next = curr
            prev = group_head
        for _ in range(k):
            if curr is None:
                break
            prev = curr
            curr = curr.next
    return head

def reverse_sublist(node, k):
    prev = None
    curr = node
    for _ in range(k):
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

## Detect Cycle (Floyd's Tortoise and Hare)

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

## Find Cycle Start Node

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

## Find Cycle Length

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

## Remove Cycle

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

## Find Nth from End (Two Pointers)

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

## Check if Palindrome (Reverse Second Half)

```python
def is_palindrome_reverse_half(head):
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
    while second is not None:
        if first.data != second.data:
            reverse_iterative(second_half)
            return False
        first = first.next
        second = second.next
    reverse_iterative(second_half)
    return True
```

## Check if Palindrome (Stack)

```python
def is_palindrome_stack(head):
    stack = []
    curr = head
    while curr is not None:
        stack.append(curr.data)
        curr = curr.next
    curr = head
    while curr is not None:
        if curr.data != stack.pop():
            return False
        curr = curr.next
    return True
```

## Check if Palindrome (Recursive)

```python
def is_palindrome_recursive(head):
    def check(node):
        nonlocal front
        if node is None:
            return True
        if not check(node.next):
            return False
        if front.data != node.data:
            return False
        front = front.next
        return True
    front = head
    return check(head)
```

## Remove Duplicates from Sorted

```python
def remove_duplicates_sorted(head):
    curr = head
    while curr is not None and curr.next is not None:
        if curr.data == curr.next.data:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return head
```

## Remove Duplicates from Unsorted (Hashing)

```python
def remove_duplicates_unsorted(head):
    seen = set()
    prev = None
    curr = head
    while curr is not None:
        if curr.data in seen:
            prev.next = curr.next
        else:
            seen.add(curr.data)
            prev = curr
        curr = curr.next
    return head
```

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

## Split List into Two Halves

```python
def split_list(head):
    if head is None:
        return None, None
    slow = head
    fast = head
    prev = None
    while fast is not None and fast.next is not None:
        prev = slow
        slow = slow.next
        fast = fast.next.next
    if prev is not None:
        prev.next = None
    first = head
    second = slow
    return first, second
```

## Sort Linked List (Merge Sort O(n log n))

```python
def merge_sort_list(head):
    if head is None or head.next is None:
        return head
    mid = find_middle(head)
    second = mid.next
    mid.next = None
    left = merge_sort_list(head)
    right = merge_sort_list(second)
    return merge_sorted_iterative(left, right)
```

## Intersection Point of Two Lists

```python
def get_intersection_node(headA, headB):
    lenA = count_nodes_iterative(headA)
    lenB = count_nodes_iterative(headB)
    currA = headA
    currB = headB
    if lenA > lenB:
        for _ in range(lenA - lenB):
            currA = currA.next
    else:
        for _ in range(lenB - lenA):
            currB = currB.next
    while currA is not None and currB is not None:
        if currA == currB:
            return currA
        currA = currA.next
        currB = currB.next
    return None
```

## Union of Two Sorted Lists

```python
def union_sorted(l1, l2):
    dummy = Node(0)
    tail = dummy
    while l1 is not None and l2 is not None:
        if l1.data < l2.data:
            tail.next = Node(l1.data)
            l1 = l1.next
        elif l2.data < l1.data:
            tail.next = Node(l2.data)
            l2 = l2.next
        else:
            tail.next = Node(l1.data)
            l1 = l1.next
            l2 = l2.next
        tail = tail.next
    while l1 is not None:
        tail.next = Node(l1.data)
        tail = tail.next
        l1 = l1.next
    while l2 is not None:
        tail.next = Node(l2.data)
        tail = tail.next
        l2 = l2.next
    return dummy.next
```

## Intersection of Two Sorted Lists

```python
def intersection_sorted(l1, l2):
    dummy = Node(0)
    tail = dummy
    while l1 is not None and l2 is not None:
        if l1.data == l2.data:
            tail.next = Node(l1.data)
            tail = tail.next
            l1 = l1.next
            l2 = l2.next
        elif l1.data < l2.data:
            l1 = l1.next
        else:
            l2 = l2.next
    return dummy.next
```

## Flatten Multilevel Linked List

```python
class MultiNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.child = None

def flatten_multilevel(head):
    if head is None:
        return None
    tail = head
    while tail.next is not None:
        tail = tail.next
    curr = head
    while curr is not None:
        if curr.child is not None:
            tail.next = curr.child
            while tail.next is not None:
                tail = tail.next
            curr.child = None
        curr = curr.next
    return head
```

## Clone List with Random Pointer (Hashmap)

```python
class RandomNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.random = None

def clone_random_hashmap(head):
    if head is None:
        return None
    mapping = {}
    curr = head
    while curr is not None:
        mapping[curr] = RandomNode(curr.data)
        curr = curr.next
    curr = head
    while curr is not None:
        mapping[curr].next = mapping.get(curr.next)
        mapping[curr].random = mapping.get(curr.random)
        curr = curr.next
    return mapping[head]
```

## Clone List with Random Pointer (Interleaving)

```python
def clone_random_interleaving(head):
    if head is None:
        return None
    curr = head
    while curr is not None:
        new_node = RandomNode(curr.data)
        new_node.next = curr.next
        curr.next = new_node
        curr = new_node.next
    curr = head
    while curr is not None:
        if curr.random is not None:
            curr.next.random = curr.random.next
        curr = curr.next.next
    curr = head
    clone_head = head.next
    while curr is not None:
        temp = curr.next
        curr.next = temp.next
        if temp.next is not None:
            temp.next = temp.next.next
        curr = curr.next
    return clone_head
```

## Rotate List by k

```python
def rotate_list(head, k):
    if head is None or head.next is None or k == 0:
        return head
    n = count_nodes_iterative(head)
    k = k % n
    if k == 0:
        return head
    fast = head
    for _ in range(k):
        fast = fast.next
    slow = head
    while fast.next is not None:
        slow = slow.next
        fast = fast.next
    new_head = slow.next
    slow.next = None
    fast.next = head
    return new_head
```

## Swap Nodes in Pairs

```python
def swap_pairs(head):
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

## Swap Kth from Beginning with Kth from End

```python
def swap_kth_nodes(head, k):
    n = count_nodes_iterative(head)
    if k > n or 2 * k - 1 == n:
        return head
    first_prev = None
    first = head
    for _ in range(k - 1):
        first_prev = first
        first = first.next
    second_prev = None
    second = head
    for _ in range(n - k):
        second_prev = second
        second = second.next
    if first_prev is not None:
        first_prev.next = second
    if second_prev is not None:
        second_prev.next = first
    temp = first.next
    first.next = second.next
    second.next = temp
    if k == 1:
        head = second
    if k == n:
        head = first
    return head
```

## Add Two Numbers as Lists

```python
def add_two_numbers(l1, l2):
    dummy = Node(0)
    curr = dummy
    carry = 0
    while l1 is not None or l2 is not None or carry:
        v1 = l1.data if l1 else 0
        v2 = l2.data if l2 else 0
        total = v1 + v2 + carry
        carry = total // 10
        curr.next = Node(total % 10)
        curr = curr.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next
```

## Segregate Even and Odd Nodes

```python
def segregate_even_odd(head):
    even_head = Node(0)
    odd_head = Node(0)
    even_tail = even_head
    odd_tail = odd_head
    curr = head
    while curr is not None:
        if curr.data % 2 == 0:
            even_tail.next = curr
            even_tail = even_tail.next
        else:
            odd_tail.next = curr
            odd_tail = odd_tail.next
        curr = curr.next
    even_tail.next = odd_head.next
    odd_tail.next = None
    return even_head.next
```

## Delete Node with Only Pointer to That Node

```python
def delete_node_without_head(node):
    if node is None or node.next is None:
        return
    node.data = node.next.data
    node.next = node.next.next
```

## LRU Cache Implementation

```python
class LRUNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = LRUNode(0, 0)
        self.tail = LRUNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add(node)
        return node.value

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = LRUNode(key, value)
        self._add(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

## LFU Cache Implementation

```python
from collections import defaultdict

class LFUNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.freq = 1
        self.prev = None
        self.next = None

class FreqList:
    def __init__(self):
        self.head = LFUNode(0, 0)
        self.tail = LFUNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def add(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def is_empty(self):
        return self.head.next == self.tail

class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.freq_map = defaultdict(FreqList)
        self.min_freq = 0

    def get(self, key):
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self.freq_map[node.freq].remove(node)
        if self.freq_map[node.freq].is_empty() and self.min_freq == node.freq:
            self.min_freq += 1
        node.freq += 1
        self.freq_map[node.freq].add(node)
        return node.value

    def put(self, key, value):
        if self.capacity == 0:
            return
        if key in self.cache:
            self.cache[key].value = value
            self.get(key)
            return
        if len(self.cache) >= self.capacity:
            fl = self.freq_map[self.min_freq]
            lfu = fl.tail.prev
            fl.remove(lfu)
            del self.cache[lfu.key]
        node = LFUNode(key, value)
        self.cache[key] = node
        self.min_freq = 1
        self.freq_map[1].add(node)
```
