# Easy Linked List Problems

## 1. Reverse Linked List

Reverse a singly linked list. Iterative three-pointer (prev, curr, next) or recursive. Change each node's next to point to previous.

```python
def reverseList(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev
```

Time: O(n) | Space: O(1)

---

## 2. Merge Two Sorted Lists

Merge two sorted linked lists into one sorted list. Two pointers, compare and link smaller node. Append remainder when one list exhausts.

```python
def mergeTwoLists(l1, l2):
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

Time: O(n+m) | Space: O(1)

---

## 3. Linked List Cycle

Determine if a linked list has a cycle. Floyd's tortoise and hare. Slow and fast pointers; if they meet, cycle exists.

```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

Time: O(n) | Space: O(1)

---

## 4. Remove Duplicates from Sorted List

Delete all duplicates such that each element appears only once. Single pass. If curr.data == curr.next.data, skip curr.next. Otherwise advance.

```python
def deleteDuplicates(head):
    cur = head
    while cur and cur.next:
        if cur.val == cur.next.val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return head
```

Time: O(n) | Space: O(1)

---

## 5. Remove Linked List Elements

Remove all nodes with a given value. Handle head deletions first with a loop. Then traverse and skip nodes with target value.

```python
def removeElements(head, val):
    dummy = ListNode(0)
    dummy.next = head
    cur = dummy
    while cur.next:
        if cur.next.val == val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next
```

Time: O(n) | Space: O(1)

---

## 6. Palindrome Linked List

Check if a linked list is a palindrome. Find middle with slow-fast, reverse second half, compare both halves. Restore second half.

```python
def isPalindrome(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    prev = None
    while slow:
        nxt = slow.next
        slow.next = prev
        prev = slow
        slow = nxt
    while prev:
        if head.val != prev.val:
            return False
        head, prev = head.next, prev.next
    return True
```

Time: O(n) | Space: O(1)

---

## 7. Middle of the Linked List

Return the middle node. If two middles, return the second. Slow-fast pointers. When fast reaches end, slow is at middle.

```python
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

Time: O(n) | Space: O(1)

---

## 8. Delete Node in a Linked List

Delete a node given only a pointer to that node (not the head). Copy next node's data to current, then delete next node.

```python
def deleteNode(node):
    node.val = node.next.val
    node.next = node.next.next
```

Time: O(1) | Space: O(1)

---

## 9. Remove Nth Node From End of List

Remove the nth node from the end in one pass. Dummy node. Lead pointer n+1 steps ahead. When lead reaches end, trailing pointer's next is the node to remove.

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

## 10. Convert Binary Number in a Linked List to Integer

Each node holds 0 or 1. Return decimal value of the binary number. Traverse and accumulate: result = result * 2 + node.val.

```python
def getDecimalValue(head):
    ans = 0
    while head:
        ans = ans * 2 + head.val
        head = head.next
    return ans
```

Time: O(n) | Space: O(1)

---

## 11. Design Linked List

Implement get, addAtHead, addAtTail, addAtIndex, deleteAtIndex. Maintain head and optionally tail. Handle edge cases for index 0 and out of bounds.

```python
class MyLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def get(self, index):
        if index < 0 or index >= self.size:
            return -1
        cur = self.head
        for _ in range(index):
            cur = cur.next
        return cur.val

    def addAtHead(self, val):
        self.addAtIndex(0, val)

    def addAtTail(self, val):
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index, val):
        if index < 0 or index > self.size:
            return
        node = ListNode(val)
        if index == 0:
            node.next = self.head
            self.head = node
        else:
            cur = self.head
            for _ in range(index - 1):
                cur = cur.next
            node.next = cur.next
            cur.next = node
        self.size += 1

    def deleteAtIndex(self, index):
        if index < 0 or index >= self.size:
            return
        if index == 0:
            self.head = self.head.next
        else:
            cur = self.head
            for _ in range(index - 1):
                cur = cur.next
            cur.next = cur.next.next
        self.size -= 1
```

Time: O(1) getHead/Tail, O(k) get/add/delete at index | Space: O(1) per operation

---

## 12. Merge Two Sorted Lists (In-Place)

Merge two sorted lists without creating new nodes. Use one list as base, insert nodes from the other in correct position. Or use dummy and relink.

```python
def mergeTwoListsInPlace(l1, l2):
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

Time: O(n+m) | Space: O(1)

---

## 13. Intersection of Two Linked Lists

Find the node where two lists intersect (by reference). Return null if no intersection. Find lengths, advance longer list by difference, traverse both in parallel until same node.

```python
def getIntersectionNode(headA, headB):
    a, b = headA, headB
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA
    return a
```

Time: O(n+m) | Space: O(1)

---

## 14. Reverse Linked List II

Reverse nodes from position m to n (1-indexed). Reach node before m, reverse m to n using one-pass reversal, reconnect.

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

## 15. Swap Nodes in Pairs

Swap every two adjacent nodes. Dummy node. For each pair, swap first and second. Advance by two.

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

## 16. Add Two Numbers

Two lists represent numbers in reverse (LSB first). Return sum as a list. Add digit by digit, maintain carry. Create new nodes for result.

```python
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    cur = dummy
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        s = v1 + v2 + carry
        carry = s // 10
        cur.next = ListNode(s % 10)
        cur = cur.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummy.next
```

Time: O(n+m) | Space: O(1) excluding output

---

## 17. Remove Duplicates from Sorted List II

Remove all nodes that have duplicate values (keep only distinct values). Dummy node. Skip all nodes with same value as next. If no duplicate, add to result.

```python
def deleteDuplicates2(head):
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

## 18. Partition List

Partition list so all nodes < x come before nodes >= x. Preserve order. Two dummy lists (less, ge). Traverse and append to appropriate list. Concatenate.

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

## 19. Odd Even Linked List

Group odd-indexed nodes first, then even-indexed. Use O(1) space. Two pointers for odd and even lists. Interleave in one pass.

```python
def oddEvenList(head):
    if not head or not head.next:
        return head
    odd, even = head, head.next
    even_head = even
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    odd.next = even_head
    return head
```

Time: O(n) | Space: O(1)

---

## 20. Sort List

Sort linked list in O(n log n) time and O(1) space (excluding recursion). Merge sort. Find middle with slow-fast, split, recursively sort, merge.

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

Time: O(n log n) | Space: O(log n) recursion

---

## 21. Reorder List

L0 -> Ln -> L1 -> Ln-1 -> L2 -> ... Find middle, reverse second half, interleave first and reversed second.

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

## 22. Copy List with Random Pointer

Deep copy a list where each node has next and random pointer. HashMap: old node -> new node. Two passes: create nodes, set next and random.

```python
def copyRandomList(head):
    m = {}
    cur = head
    while cur:
        m[cur] = Node(cur.val)
        cur = cur.next
    cur = head
    while cur:
        m[cur].next = m.get(cur.next)
        m[cur].random = m.get(cur.random)
        cur = cur.next
    return m.get(head)
```

Time: O(n) | Space: O(n)

---

## 23. Linked List Cycle II

Return the node where the cycle begins. Return null if no cycle. Floyd's cycle detection. After meeting, one pointer at head, both move one step. Meeting point is cycle start.

```python
def detectCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            p = head
            while p != slow:
                p = p.next
                slow = slow.next
            return p
    return None
```

Time: O(n) | Space: O(1)

---

## 24. Flatten a Multilevel Doubly Linked List

Flatten a multilevel doubly linked list (nodes can have child pointers). DFS or iterative. When node has child, append child list to current tail, continue.

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

## 25. Design Browser History

Implement back, forward, visit for browser history using doubly linked list. Doubly linked list with current pointer. Visit clears forward history.

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

Time: O(steps) per back/forward | Space: O(n)
