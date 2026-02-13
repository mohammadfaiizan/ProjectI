# Medium Linked List Problems

## 1. Add Two Numbers II

Two non-empty lists represent numbers (MSB first). Return sum as a list without reversing. Use stacks to reverse digit order, then add. Or reverse both lists, add, reverse result.

```python
def addTwoNumbers(l1, l2):
    s1, s2 = [], []
    while l1:
        s1.append(l1.val)
        l1 = l1.next
    while l2:
        s2.append(l2.val)
        l2 = l2.next
    carry = 0
    head = None
    while s1 or s2 or carry:
        v1 = s1.pop() if s1 else 0
        v2 = s2.pop() if s2 else 0
        s = v1 + v2 + carry
        carry = s // 10
        node = ListNode(s % 10)
        node.next = head
        head = node
    return head
```

Time: O(n+m) | Space: O(n+m)

---

## 2. Swap Nodes in Pairs

Swap every two adjacent nodes. Must modify in-place. Dummy node. For each pair, reverse the two nodes and link to previous group.

```python
def swapPairs(head):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    while prev.next and prev.next.next:
        first = prev.next
        second = first.next
        prev.next = second
        first.next = second.next
        second.next = first
        prev = first
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 3. Remove Nth Node From End of List

Remove nth node from end in one pass. Lead pointer n+1 steps ahead. When lead reaches null, trailing's next is target.

```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0)
    dummy.next = head
    lead = trail = dummy
    for _ in range(n + 1):
        lead = lead.next
    while lead:
        lead = lead.next
        trail = trail.next
    trail.next = trail.next.next
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 4. Rotate List

Rotate list to the right by k places. k = k % n. Find (n-k)th node from start. That node becomes new tail. Its next becomes new head.

```python
def rotateRight(head, k):
    if not head or not head.next:
        return head
    n, tail = 1, head
    while tail.next:
        tail = tail.next
        n += 1
    k %= n
    if k == 0:
        return head
    cur = head
    for _ in range(n - k - 1):
        cur = cur.next
    new_head = cur.next
    cur.next = None
    tail.next = head
    return new_head
```

Time: O(n) | Space: O(1)

---

## 5. Reverse Nodes in k-Group

Reverse every k consecutive nodes. Remainder stays as is. Recursive or iterative. For each group of k, reverse and connect. Handle remainder.

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

Time: O(n) | Space: O(n/k) recursion

---

## 6. Flatten a Multilevel Doubly Linked List

Flatten multilevel list (child pointers) into single-level doubly linked list. Iterative. When node has child, find tail of current list, append child, set child's prev. Continue.

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

## 7. Copy List with Random Pointer

Deep copy list with next and random pointers. Interleaving method - insert copy after each node, set random, extract copies. Or hashmap.

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

## 8. Reorder List

L0 -> Ln -> L1 -> Ln-1 -> L2 -> ... Find middle (slow-fast), reverse second half, merge by alternating nodes.

```python
def reorderList(head):
    if not head or not head.next:
        return
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    prev, cur = None, slow.next
    slow.next = None
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    first, second = head, prev
    while second:
        t1, t2 = first.next, second.next
        first.next = second
        second.next = t1
        first, second = t1, t2
```

Time: O(n) | Space: O(1)

---

## 9. Sort List

Sort in O(n log n) time, O(1) space (or O(log n) recursion stack). Merge sort. Split at middle, sort halves, merge.

```python
def sortList(head):
    if not head or not head.next:
        return head
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None
    left = sortList(head)
    right = sortList(mid)
    dummy = ListNode(0)
    cur = dummy
    while left and right:
        if left.val <= right.val:
            cur.next = left
            left = left.next
        else:
            cur.next = right
            right = right.next
        cur = cur.next
    cur.next = left or right
    return dummy.next
```

Time: O(n log n) | Space: O(log n)

---

## 10. Insertion Sort List

Sort list using insertion sort. Build sorted list incrementally. For each node, find position in sorted prefix and insert.

```python
def insertionSortList(head):
    dummy = ListNode(float('-inf'))
    while head:
        prev = dummy
        while prev.next and prev.next.val < head.val:
            prev = prev.next
        nxt = head.next
        head.next = prev.next
        prev.next = head
        head = nxt
    return dummy.next
```

Time: O(n^2) | Space: O(1)

---

## 11. Partition List

Partition so all nodes < x come before nodes >= x. Preserve relative order. Two lists (less, ge). Traverse, append to appropriate list. Concatenate.

```python
def partition(head, x):
    less = less_head = ListNode(0)
    ge = ge_head = ListNode(0)
    while head:
        if head.val < x:
            less.next = head
            less = less.next
        else:
            ge.next = head
            ge = ge.next
        head = head.next
    ge.next = None
    less.next = ge_head.next
    return less_head.next
```

Time: O(n) | Space: O(1)

---

## 12. Remove Duplicates from Sorted List II

Delete all nodes that have duplicates. Keep only distinct values. Dummy. While curr.next exists, skip all nodes equal to curr.next. If no skip happened, add curr to result.

```python
def deleteDuplicates(head):
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    while prev.next:
        cur = prev.next
        while cur.next and cur.next.val == cur.val:
            cur = cur.next
        if cur != prev.next:
            prev.next = cur.next
        else:
            prev = prev.next
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 13. Reverse Linked List II

Reverse from position m to n (1-indexed). Reach node before m. Reverse m to n in one pass (repeatedly move curr.next to front of sublist).

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

## 14. Linked List Random Node

Return random node with equal probability. List length unknown. Reservoir sampling. For each node i, replace result with probability 1/i.

```python
def __init__(self, head):
    self.head = head

def getRandom(self):
    cur = self.head
    res = cur.val
    i = 1
    while cur.next:
        cur = cur.next
        i += 1
        if random.randint(1, i) == 1:
            res = cur.val
    return res
```

Time: O(n) per getRandom | Space: O(1)

---

## 15. Split Linked List in Parts

Split list into k consecutive parts. Lengths should differ by at most 1. Count n. Base size = n // k, remainder = n % k. First remainder parts get base+1 nodes.

```python
def splitListToParts(head, k):
    n = 0
    cur = head
    while cur:
        n += 1
        cur = cur.next
    base, rem = n // k, n % k
    res = []
    cur = head
    for i in range(k):
        part_head = cur
        size = base + (1 if i < rem else 0)
        for _ in range(size - 1):
            if cur:
                cur = cur.next
        if cur:
            nxt = cur.next
            cur.next = None
            cur = nxt
        res.append(part_head)
    return res
```

Time: O(n) | Space: O(1)

---

## 16. Next Greater Node In Linked List

For each node, find next greater value to the right. Store in array. Convert to array, use monotonic stack to find next greater for each index.

```python
def nextLargerNodes(head):
    arr = []
    while head:
        arr.append(head.val)
        head = head.next
    st = []
    res = [0] * len(arr)
    for i in range(len(arr)):
        while st and arr[st[-1]] < arr[i]:
            res[st.pop()] = arr[i]
        st.append(i)
    return res
```

Time: O(n) | Space: O(n)

---

## 17. Remove Zero Sum Consecutive Nodes

Remove sequences of nodes that sum to zero. Prefix sum + hashmap. If prefix_sum seen before, remove nodes between. Repeat until no change.

```python
def removeZeroSumSublists(head):
    dummy = ListNode(0)
    dummy.next = head
    prefix = 0
    seen = {0: dummy}
    cur = head
    while cur:
        prefix += cur.val
        seen[prefix] = cur
        cur = cur.next
    prefix = 0
    cur = dummy
    while cur:
        prefix += cur.val
        cur.next = seen[prefix].next
        cur = cur.next
    return dummy.next
```

Time: O(n) | Space: O(n)

---

## 18. Design Linked List (Doubly)

Implement doubly linked list with get, addAtHead, addAtTail, addAtIndex, deleteAtIndex. Maintain head and tail. Update prev and next for all operations.

```python
class MyLinkedList:
    def __init__(self):
        self.head = self.tail = None
        self.size = 0

    def get(self, index):
        if index < 0 or index >= self.size:
            return -1
        cur = self.head
        for _ in range(index):
            cur = cur.next
        return cur.val

    def addAtHead(self, val):
        node = DNode(val)
        if self.size == 0:
            self.head = self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
        self.size += 1

    def addAtTail(self, val):
        node = DNode(val)
        if self.size == 0:
            self.head = self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
        self.size += 1

    def addAtIndex(self, index, val):
        if index < 0 or index > self.size:
            return
        if index == 0:
            self.addAtHead(val)
        elif index == self.size:
            self.addAtTail(val)
        else:
            cur = self.head
            for _ in range(index):
                cur = cur.next
            node = DNode(val)
            node.next = cur
            node.prev = cur.prev
            cur.prev.next = node
            cur.prev = node
            self.size += 1

    def deleteAtIndex(self, index):
        if index < 0 or index >= self.size:
            return
        cur = self.head
        for _ in range(index):
            cur = cur.next
        if cur.prev:
            cur.prev.next = cur.next
        else:
            self.head = cur.next
        if cur.next:
            cur.next.prev = cur.prev
        else:
            self.tail = cur.prev
        self.size -= 1
```

Time: O(k) for get/add/delete at index | Space: O(1)

---

## 19. Delete the Middle Node of a Linked List

Delete the middle node (slow-fast to find, then delete). Slow-fast to find middle. Need prev of middle to delete. Use dummy or track prev.

```python
def deleteMiddle(head):
    if not head or not head.next:
        return None
    dummy = ListNode(0)
    dummy.next = head
    slow, fast = dummy, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    slow.next = slow.next.next
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 20. Maximum Twin Sum of a Linked List

Twin of node i is node n-1-i. Find maximum sum of (node + twin). Find middle, reverse second half. Traverse first and reversed second in parallel, track max sum.

```python
def pairSum(head):
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    prev, cur = None, slow.next
    slow.next = None
    while cur:
        nxt = cur.next
        cur.next = prev
        prev = cur
        cur = nxt
    res = 0
    while prev:
        res = max(res, head.val + prev.val)
        head = head.next
        prev = prev.next
    return res
```

Time: O(n) | Space: O(1)

---

## 21. Delete Nodes and Return Forest

Given list and to_delete set, return list of roots of remaining trees (forest). Track parent. When deleting, add children to result if not in to_delete. Handle root.

```python
def delNodes(root, to_delete):
    to_delete = set(to_delete)
    res = []
    def dfs(node, is_root):
        if not node:
            return None
        deleted = node.val in to_delete
        if is_root and not deleted:
            res.append(node)
        node.left = dfs(node.left, deleted)
        node.right = dfs(node.right, deleted)
        return None if deleted else node
    dfs(root, True)
    return res
```

Time: O(n) | Space: O(h)

---

## 22. Split Circular Linked List

Split circular list into two circular lists of roughly equal size. Slow-fast to find mid. First: head to mid (circular). Second: mid.next to end (circular).

```python
def splitCircular(head):
    if not head or not head.next:
        return head, None
    slow, fast = head, head.next
    while fast.next != head and fast.next.next != head:
        slow = slow.next
        fast = fast.next.next
    second = slow.next
    slow.next = head
    cur = second
    while cur.next != head:
        cur = cur.next
    cur.next = second
    return head, second
```

Time: O(n) | Space: O(1)

---

## 23. Design Browser History

Implement visit, back, forward with max steps. Doubly linked list. Visit clears forward chain. Back/forward move current pointer.

```python
class BrowserHistory:
    def __init__(self, homepage):
        self.cur = Node(homepage)

    def visit(self, url):
        self.cur.next = Node(url)
        self.cur.next.prev = self.cur
        self.cur = self.cur.next

    def back(self, steps):
        for _ in range(steps):
            if self.cur.prev:
                self.cur = self.cur.prev
            else:
                break
        return self.cur.val

    def forward(self, steps):
        for _ in range(steps):
            if self.cur.next:
                self.cur = self.cur.next
            else:
                break
        return self.cur.val
```

Time: O(steps) | Space: O(n)

---

## 24. Merge In Between Linked Lists

Remove nodes from list1 between a and b (inclusive), replace with list2. Find node before a and node after b. Connect before_a to list2 head, list2 tail to after_b.

```python
def mergeInBetween(list1, a, b, list2):
    prev_a = list1
    for _ in range(a - 1):
        prev_a = prev_a.next
    node_b = prev_a
    for _ in range(b - a + 1):
        node_b = node_b.next
    prev_a.next = list2
    tail2 = list2
    while tail2.next:
        tail2 = tail2.next
    tail2.next = node_b.next
    return list1
```

Time: O(n) | Space: O(1)

---

## 25. Swapping Nodes in a Linked List

Swap kth node from beginning with kth from end. Find both nodes (nth from start: traverse k-1; nth from end: lead pointer). Swap their values or links.

```python
def swapNodes(head, k):
    lead = head
    for _ in range(k - 1):
        lead = lead.next
    first = lead
    trail = head
    while lead.next:
        lead = lead.next
        trail = trail.next
    first.val, trail.val = trail.val, first.val
    return head
```

Time: O(n) | Space: O(1)
