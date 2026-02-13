# Hard Linked List Problems

## 1. Merge k Sorted Lists

Merge k sorted linked lists into one sorted list. Min-heap of size k. Push first node of each list. Pop min, append to result, push its next. O(n log k).

```python
def mergeKLists(lists):
    import heapq
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    dummy = ListNode(0)
    cur = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```

Time: O(n log k) | Space: O(k)

---

## 2. Reverse Nodes in k-Group

Reverse every k nodes. If remaining < k, leave unchanged. Iterative: find kth node, reverse sublist, reconnect. Repeat. Or recursive: reverse first k, recurse on rest.

```python
def reverseKGroup(head, k):
    cur = head
    for _ in range(k):
        if not cur:
            return head
        cur = cur.next
    prev = None
    cur = head
    for _ in range(k):
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    head.next = reverseKGroup(cur, k)
    return prev
```

Time: O(n) | Space: O(n/k)

---

## 3. Copy List with Random Pointer

Deep copy list with next and random. No extra space for hashmap (O(1) space variant). Interleaving: insert copy after each original. Set copy.random = original.random.next. Extract copies.

```python
def copyRandomList(head):
    if not head:
        return None
    cur = head
    while cur:
        copy = Node(cur.val)
        copy.next = cur.next
        cur.next = copy
        cur = copy.next
    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
        cur = cur.next.next
    dummy = Node(0)
    copy_cur = dummy
    cur = head
    while cur:
        copy_cur.next = cur.next
        cur.next = cur.next.next
        copy_cur = copy_cur.next
        cur = cur.next
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 4. Merge Two Sorted Lists (In-Place, O(1) Space)

Merge two sorted lists using only O(1) extra space. Use one list as base. For each node in other list, find insertion position and insert. No new nodes.

```python
def mergeInPlace(l1, l2):
    dummy = ListNode(0)
    dummy.next = l1
    prev = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            prev = l1
            l1 = l1.next
        else:
            nxt = l2.next
            prev.next = l2
            l2.next = l1
            prev = l2
            l2 = nxt
    if l2:
        prev.next = l2
    return dummy.next
```

Time: O(n+m) | Space: O(1)

---

## 5. Reverse Linked List (Between m and n, One-Pass)

Reverse nodes from position m to n in one pass. Reach node before m. Repeatedly move curr.next to front of reversed sublist. Do this (n-m) times.

```python
def reverseBetween(head, m, n):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    for _ in range(m - 1):
        prev = prev.next
    cur = prev.next
    for _ in range(n - m):
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = prev.next
        prev.next = nxt
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 6. Flatten a Multilevel Doubly Linked List

Flatten multilevel list. Each node has next, prev, and child. DFS order. Recursive flatten. When node has child, flatten child, insert between node and node.next.

```python
def flatten(head):
    cur = head
    while cur:
        if cur.child:
            nxt = cur.next
            child = cur.child
            cur.next = child
            child.prev = cur
            cur.child = None
            tail = child
            while tail.next:
                tail = tail.next
            tail.next = nxt
            if nxt:
                nxt.prev = tail
        cur = cur.next
    return head
```

Time: O(n) | Space: O(1)

---

## 7. LRU Cache

Implement LRU cache with get and put in O(1). Evict least recently used when capacity exceeded. HashMap for O(1) lookup. Doubly linked list for O(1) move-to-front and remove-last. On get: move to front. On put: add to front, evict last if full.

```python
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

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

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add(self, node):
        self.tail.prev.next = node
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev = node
```

Time: O(1) get/put | Space: O(capacity)

---

## 8. LFU Cache

Implement LFU cache. Evict least frequently used. On tie, evict LRU. HashMap key->node. HashMap freq->doubly linked list of nodes. Track min_freq. On eviction, remove from min_freq list.

```python
class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.min_freq = 0
        self.key_to_node = {}
        self.freq_to_list = defaultdict(lambda: DoublyLinkedList())

    def get(self, key):
        if key not in self.key_to_node:
            return -1
        node = self.key_to_node[key]
        self._update(node)
        return node.val

    def put(self, key, value):
        if self.cap == 0:
            return
        if key in self.key_to_node:
            node = self.key_to_node[key]
            node.val = value
            self._update(node)
            return
        if len(self.key_to_node) >= self.cap:
            lst = self.freq_to_list[self.min_freq]
            lru = lst.pop_tail()
            del self.key_to_node[lru.key]
        node = Node(key, value, 1)
        self.freq_to_list[1].add_head(node)
        self.key_to_node[key] = node
        self.min_freq = 1

    def _update(self, node):
        freq = node.freq
        self.freq_to_list[freq].remove(node)
        if self.min_freq == freq and self.freq_to_list[freq].is_empty():
            self.min_freq += 1
        node.freq += 1
        self.freq_to_list[node.freq].add_head(node)
```

Time: O(1) get/put | Space: O(capacity)

---

## 9. All O one Data Structure

Implement inc, dec, getMaxKey, getMinKey all in O(1). HashMap key->count. HashMap count->set of keys. Doubly linked list of counts for max/min. On inc/dec, move key between count buckets.

```python
class AllOne:
    def __init__(self):
        self.key_count = {}
        self.count_keys = defaultdict(set)
        self.order = []
        self.min_count = float('inf')
        self.max_count = 0

    def inc(self, key):
        old = self.key_count.get(key, 0)
        new = old + 1
        self.key_count[key] = new
        if old:
            self.count_keys[old].discard(key)
        self.count_keys[new].add(key)
        self.max_count = max(self.max_count, new)
        if self.min_count == old and not self.count_keys[old]:
            self.min_count = new if old == 0 else min(k for k in self.count_keys if self.count_keys[k])

    def dec(self, key):
        old = self.key_count[key]
        new = old - 1
        if new:
            self.key_count[key] = new
            self.count_keys[new].add(key)
        else:
            del self.key_count[key]
        self.count_keys[old].discard(key)
        if not self.count_keys[old]:
            del self.count_keys[old]

    def getMaxKey(self):
        if not self.key_count:
            return ""
        for c in range(self.max_count, 0, -1):
            if c in self.count_keys and self.count_keys[c]:
                return next(iter(self.count_keys[c]))
        return ""

    def getMinKey(self):
        if not self.key_count:
            return ""
        for c in range(1, self.max_count + 1):
            if c in self.count_keys and self.count_keys[c]:
                return next(iter(self.count_keys[c]))
        return ""
```

Time: O(1) amortized | Space: O(n)

---

## 10. Design Skiplist

Implement skiplist with search, add, erase. Probabilistic multi-level linked structure. Each node has multiple forward pointers. Levels determined by random. Search: start at top level, go right while next < target, else go down.

```python
import random
class Skiplist:
    def __init__(self):
        self.head = [None] * 16

    def _iter(self, num):
        cur = self.head
        for level in range(15, -1, -1):
            while cur[level] and cur[level].val < num:
                cur = cur[level]
            yield cur, level

    def search(self, target):
        for prev, level in self._iter(target):
            pass
        cur = prev[0]
        return cur and cur.val == target

    def add(self, num):
        node = [None] * 16
        node[0] = num
        for prev, level in self._iter(num):
            if level < 16 and random.random() < 0.5:
                node[level] = prev[level]
                prev[level] = node

    def erase(self, num):
        found = False
        for prev, level in self._iter(num):
            nxt = prev[level]
            if nxt and nxt.val == num:
                found = True
                prev[level] = nxt[level]
        return found
```

Time: O(log n) expected | Space: O(n)

---

## 11. Find the Minimum and Maximum Number of Nodes Between Critical Points

Critical point: local max or min. Find min and max distance between consecutive critical points. Traverse, identify critical points (compare with prev and next). Store indices. Min: consecutive difference. Max: first to last.

```python
def nodesBetweenCriticalPoints(head):
    crit = []
    prev_val = None
    i = 0
    while head and head.next:
        if prev_val is not None and head.next:
            if (head.val > prev_val and head.val > head.next.val) or (head.val < prev_val and head.val < head.next.val):
                crit.append(i)
        prev_val = head.val
        head = head.next
        i += 1
    if len(crit) < 2:
        return [-1, -1]
    min_d = min(crit[j] - crit[j-1] for j in range(1, len(crit)))
    max_d = crit[-1] - crit[0]
    return [min_d, max_d]
```

Time: O(n) | Space: O(n)

---

## 12. Reverse Alternating k-Group

Reverse first k nodes, skip next k, reverse next k, skip, etc. For each group: if reverse group, reverse k nodes and connect. If skip group, just advance k nodes. Alternate flag.

```python
def reverseAlternatingKGroup(head, k):
    def reverse(first, n):
        prev = None
        cur = first
        for _ in range(n):
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev, cur

    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    do_reverse = True
    while prev.next:
        cur = prev.next
        for _ in range(k - 1):
            if not cur.next:
                return dummy.next
            cur = cur.next
        nxt_start = cur.next
        if do_reverse:
            rev_head, rev_tail_next = reverse(prev.next, k)
            prev.next = rev_head
            rev_tail = prev.next
            for _ in range(k - 1):
                rev_tail = rev_tail.next
            rev_tail.next = nxt_start
            prev = rev_tail
        else:
            prev = cur
        do_reverse = not do_reverse
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 13. Sort a Linked List of 0s 1s 2s

Sort list containing only 0, 1, 2. One pass, O(1) space. Three dummy lists. Traverse once, append each node to correct list. Concatenate 0, 1, 2.

```python
def sort012(head):
    zero = z = ListNode(0)
    one = o = ListNode(0)
    two = t = ListNode(0)
    while head:
        if head.val == 0:
            z.next = head
            z = z.next
        elif head.val == 1:
            o.next = head
            o = o.next
        else:
            t.next = head
            t = t.next
        head = head.next
    z.next = one.next
    o.next = two.next
    t.next = None
    return zero.next
```

Time: O(n) | Space: O(1)

---

## 14. Flatten a Multilevel Linked List (Depth-First)

Flatten so child lists come before next sibling. DFS order. Recursive. Process node, then child (recursive), then next. Build result during recursion.

```python
def flatten(head):
    cur = head
    while cur:
        if cur.child:
            nxt = cur.next
            child = cur.child
            cur.next = child
            child.prev = cur
            cur.child = None
            tail = child
            while tail.next:
                tail = tail.next
            tail.next = nxt
            if nxt:
                nxt.prev = tail
        cur = cur.next
    return head
```

Time: O(n) | Space: O(1)

---

## 15. Clone a Linked List with Next and Random Pointer (O(1) Space)

Clone list with random pointer without hashmap. Interleaving. Create copy after each node. Set copy.random = original.random.next. Split into two lists.

```python
def copyRandomList(head):
    if not head:
        return None
    cur = head
    while cur:
        copy = Node(cur.val)
        copy.next = cur.next
        cur.next = copy
        cur = copy.next
    cur = head
    while cur:
        if cur.random:
            cur.next.random = cur.random.next
        cur = cur.next.next
    dummy = Node(0)
    copy_cur = dummy
    cur = head
    while cur:
        copy_cur.next = cur.next
        cur.next = cur.next.next
        copy_cur = copy_cur.next
        cur = cur.next
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 16. Merge k Sorted Lists (Divide and Conquer)

Merge k lists in O(n log k) using divide and conquer. Pair up lists, merge pairs. Repeat until one list. Each level processes all n nodes. Log k levels.

```python
def mergeKLists(lists):
    if not lists:
        return None
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged.append(mergeTwo(l1, l2))
        lists = merged
    return lists[0]

def mergeTwo(l1, l2):
    dummy = ListNode(0)
    cur = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next
```

Time: O(n log k) | Space: O(1)

---

## 17. Reverse a Linked List in Groups of K (Alternating)

First k reversed, next k as-is, next k reversed, etc. Track whether to reverse. For reverse: reverse k nodes. For skip: advance k nodes. Toggle flag.

```python
def reverseAlternatingKGroup(head, k):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    do_reverse = True
    while prev.next:
        cur = prev.next
        for _ in range(k - 1):
            if not cur.next:
                return dummy.next
            cur = cur.next
        nxt_start = cur.next
        if do_reverse:
            h, t = prev.next, prev.next
            for _ in range(k - 1):
                n = t.next
                t.next = n.next
                n.next = h
                h = n
            prev.next = h
            while t.next != nxt_start:
                t = t.next
            prev = t
        else:
            prev = cur
        do_reverse = not do_reverse
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 18. Flatten a Multilevel Doubly Linked List (Level-Order)

Flatten in level order (BFS): all level 0, then all level 1, etc. Queue. Process node, add next and child to queue. Build result in BFS order.

```python
def flatten(head):
    from collections import deque
    q = deque()
    cur = head
    while cur:
        if cur.child:
            q.append(cur.child)
            cur.child = None
        if cur.next:
            cur = cur.next
        elif q:
            child = q.popleft()
            cur.next = child
            child.prev = cur
            cur = child
        else:
            break
    return head
```

Time: O(n) | Space: O(n)
