# Easy Queue Problems

## 1. Implement Queue using Stacks

Implement a FIFO queue using only two stacks. The queue should support push, pop, peek, and empty operations. Use one stack for enqueue. For dequeue/peek, transfer all elements to a second stack so the bottom of the first becomes the top of the second. Pop from the second stack for dequeue. Amortized O(1) per operation.

```python
class MyQueue:
    def __init__(self):
        self.in_st = []
        self.out_st = []

    def push(self, x):
        self.in_st.append(x)

    def pop(self):
        self._transfer()
        return self.out_st.pop()

    def peek(self):
        self._transfer()
        return self.out_st[-1]

    def empty(self):
        return not self.in_st and not self.out_st

    def _transfer(self):
        if not self.out_st:
            while self.in_st:
                self.out_st.append(self.in_st.pop())
```

Time: O(1) amortized | Space: O(n)

---

## 2. Implement Stack using Queues

Implement a LIFO stack using only one or two queues. Support push, pop, top, and empty. Single queue: push new element, then rotate n-1 elements to bring it to front. Pop is O(1). Two queues: push to q1; for pop, move n-1 from q1 to q2, pop the last from q1, swap queues.

```python
from collections import deque
class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0
```

Time: O(1) pop/top, O(n) push | Space: O(n)

---

## 3. Design Hit Counter

Design a hit counter that counts hits in the last 300 seconds. Support hit(timestamp) and getHits(timestamp). Use a queue to store timestamps. On getHits, remove timestamps older than timestamp - 300, then return queue size.

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

Time: O(1) hit, O(n) getHits | Space: O(n)

---

## 4. Number of Recent Calls

Design RecentCounter with ping(t). Return the number of requests in the last 3000 ms. Queue of timestamps. On ping, append t, remove from front while front < t - 3000, return queue size.

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

Time: O(1) amortized | Space: O(n)

---

## 5. Moving Average from Data Stream

Calculate the moving average of the last size values. Support next(val). Queue of size at most `size`. Maintain running sum. When full, dequeue oldest and subtract from sum before enqueueing new.

```python
from collections import deque
class MovingAverage:
    def __init__(self, size):
        self.q = deque()
        self.size = size
        self.total = 0

    def next(self, val):
        if len(self.q) == self.size:
            self.total -= self.q.popleft()
        self.q.append(val)
        self.total += val
        return self.total / len(self.q)
```

Time: O(1) | Space: O(size)

---

## 6. First Unique Character in a String

Find the index of the first non-repeating character in a string. Return -1 if none. Count frequency of each character. Iterate and return first with count 1. Can also use queue: enqueue chars, dequeue while front has count > 1.

```python
def firstUniqChar(s):
    from collections import Counter
    cnt = Counter(s)
    for i, c in enumerate(s):
        if cnt[c] == 1:
            return i
    return -1
```

Time: O(n) | Space: O(1)

---

## 7. Implement Queue using Stacks (Amortized O(1))

Same as problem 1; ensure amortized O(1) per operation. Stack-in for enqueue, stack-out for dequeue. When stack-out is empty, transfer all from stack-in. Each element is pushed and popped at most twice.

```python
class MyQueue:
    def __init__(self):
        self.in_st = []
        self.out_st = []

    def push(self, x):
        self.in_st.append(x)

    def pop(self):
        self._transfer()
        return self.out_st.pop()

    def peek(self):
        self._transfer()
        return self.out_st[-1]

    def empty(self):
        return not self.in_st and not self.out_st

    def _transfer(self):
        if not self.out_st:
            while self.in_st:
                self.out_st.append(self.in_st.pop())
```

Time: O(1) amortized | Space: O(n)

---

## 8. Design Circular Queue

Design a circular queue with fixed capacity. Support enQueue, deQueue, Front, Rear, isEmpty, isFull. Circular array with front and rear indices. Use modulo for wrap-around. Maintain size or sacrifice one slot to distinguish full from empty.

```python
class MyCircularQueue:
    def __init__(self, k):
        self.arr = [0] * k
        self.front = self.rear = self.size = 0
        self.k = k

    def enQueue(self, value):
        if self.isFull():
            return False
        self.arr[self.rear] = value
        self.rear = (self.rear + 1) % self.k
        self.size += 1
        return True

    def deQueue(self):
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % self.k
        self.size -= 1
        return True

    def Front(self):
        return -1 if self.isEmpty() else self.arr[self.front]

    def Rear(self):
        return -1 if self.isEmpty() else self.arr[(self.rear - 1) % self.k]

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.k
```

Time: O(1) all ops | Space: O(k)

---

## 9. Design Circular Deque

Design a double-ended queue with fixed capacity. Support insertFront, insertLast, deleteFront, deleteLast, getFront, getRear, isEmpty, isFull. Circular array. Front and rear can grow in both directions. Use modulo arithmetic for indices.

```python
class MyCircularDeque:
    def __init__(self, k):
        self.arr = [0] * k
        self.front = 0
        self.rear = 0
        self.size = 0
        self.k = k

    def insertFront(self, value):
        if self.isFull():
            return False
        self.front = (self.front - 1) % self.k
        self.arr[self.front] = value
        self.size += 1
        return True

    def insertLast(self, value):
        if self.isFull():
            return False
        self.arr[self.rear] = value
        self.rear = (self.rear + 1) % self.k
        self.size += 1
        return True

    def deleteFront(self):
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % self.k
        self.size -= 1
        return True

    def deleteLast(self):
        if self.isEmpty():
            return False
        self.rear = (self.rear - 1) % self.k
        self.size -= 1
        return True

    def getFront(self):
        return -1 if self.isEmpty() else self.arr[self.front]

    def getRear(self):
        return -1 if self.isEmpty() else self.arr[(self.rear - 1) % self.k]

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.k
```

Time: O(1) all ops | Space: O(k)

---

## 10. Generate Binary Numbers from 1 to N

Generate first n binary numbers: "1", "10", "11", "100", etc. BFS with queue. Start with "1". Dequeue, output, enqueue current+"0" and current+"1".

```python
from collections import deque
def generateBinaryNumbers(n):
    q = deque(['1'])
    res = []
    for _ in range(n):
        x = q.popleft()
        res.append(x)
        q.append(x + '0')
        q.append(x + '1')
    return res
```

Time: O(n) | Space: O(n)

---

## 11. Reverse First K Elements of Queue

Given a queue and integer k, reverse the order of the first k elements. Push first k elements to a stack. Pop back to queue. Rotate remaining n-k elements to the back.

```python
from collections import deque
def reverseFirstK(q, k):
    st = []
    for _ in range(k):
        st.append(q.popleft())
    while st:
        q.append(st.pop())
    for _ in range(len(q) - k):
        q.append(q.popleft())
    return q
```

Time: O(n) | Space: O(k)

---

## 12. Level Order Traversal of Binary Tree

Return level-order traversal (BFS) of a binary tree. Queue. Enqueue root. While queue not empty: dequeue, process, enqueue left and right children.

```python
from collections import deque
def levelOrder(root):
    if not root:
        return []
    q = deque([root])
    res = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res
```

Time: O(n) | Space: O(n)

---

## 13. Average of Levels in Binary Tree

Return the average value of nodes at each level. BFS with level tracking. For each level, sum all values and divide by count.

```python
from collections import deque
def averageOfLevels(root):
    if not root:
        return []
    q = deque([root])
    res = []
    while q:
        total, cnt = 0, len(q)
        for _ in range(cnt):
            node = q.popleft()
            total += node.val
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(total / cnt)
    return res
```

Time: O(n) | Space: O(n)

---

## 14. Minimum Depth of Binary Tree

Find the minimum depth (shortest path from root to leaf). BFS. Return depth when we first encounter a node with no children.

```python
from collections import deque
def minDepth(root):
    if not root:
        return 0
    q = deque([(root, 1)])
    while q:
        node, d = q.popleft()
        if not node.left and not node.right:
            return d
        if node.left:
            q.append((node.left, d + 1))
        if node.right:
            q.append((node.right, d + 1))
    return 0
```

Time: O(n) | Space: O(n)

---

## 15. Symmetric Tree (BFS variant)

Check if binary tree is mirror of itself. Can use BFS comparing level by level. BFS, store each level. Check if each level is palindrome. Or use two queues for left and right subtrees.

```python
from collections import deque
def isSymmetric(root):
    if not root:
        return True
    q = deque([root.left, root.right])
    while q:
        a, b = q.popleft(), q.popleft()
        if not a and not b:
            continue
        if not a or not b or a.val != b.val:
            return False
        q.extend([a.left, b.right, a.right, b.left])
    return True
```

Time: O(n) | Space: O(n)

---

## 16. Merge Two Binary Trees (BFS)

Merge two binary trees by summing overlapping nodes. Can implement with BFS. BFS both trees in parallel. When both have nodes at a position, sum values. When one is null, use the other.

```python
from collections import deque
def mergeTrees(root1, root2):
    if not root1:
        return root2
    if not root2:
        return root1
    q = deque([(root1, root2)])
    while q:
        n1, n2 = q.popleft()
        n1.val += n2.val
        if n1.left and n2.left:
            q.append((n1.left, n2.left))
        elif n2.left:
            n1.left = n2.left
        if n1.right and n2.right:
            q.append((n1.right, n2.right))
        elif n2.right:
            n1.right = n2.right
    return root1
```

Time: O(n) | Space: O(n)

---

## 17. Invert Binary Tree (BFS)

Invert a binary tree (swap left and right children). BFS implementation. BFS. For each node, swap its left and right children before enqueueing them.

```python
from collections import deque
def invertTree(root):
    if not root:
        return None
    q = deque([root])
    while q:
        node = q.popleft()
        node.left, node.right = node.right, node.left
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return root
```

Time: O(n) | Space: O(n)

---

## 18. Same Tree (BFS)

Check if two binary trees are identical. BFS comparison. BFS both trees in lockstep. Compare values at each step. Structure must match.

```python
from collections import deque
def isSameTree(p, q):
    dq = deque([(p, q)])
    while dq:
        a, b = dq.popleft()
        if not a and not b:
            continue
        if not a or not b or a.val != b.val:
            return False
        dq.append((a.left, b.left))
        dq.append((a.right, b.right))
    return True
```

Time: O(n) | Space: O(n)

---

## 19. Maximum Depth of Binary Tree (BFS)

Find the maximum depth of a binary tree. BFS variant. BFS with level counter. Increment level after processing each level. Return final level.

```python
from collections import deque
def maxDepth(root):
    if not root:
        return 0
    q = deque([root])
    depth = 0
    while q:
        for _ in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        depth += 1
    return depth
```

Time: O(n) | Space: O(n)

---

## 20. Binary Tree Level Order Traversal II

Level-order traversal but return levels from bottom to top. Standard BFS level-order, then reverse the list of levels.

```python
from collections import deque
def levelOrderBottom(root):
    if not root:
        return []
    q = deque([root])
    res = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res[::-1]
```

Time: O(n) | Space: O(n)

---

## 21. Find Bottom Left Tree Value

Find the value of the leftmost node in the last row of the tree. BFS level-order. Return the first node value of the last level.

```python
from collections import deque
def findBottomLeftValue(root):
    q = deque([root])
    while q:
        node = q.popleft()
        if node.right:
            q.append(node.right)
        if node.left:
            q.append(node.left)
    return node.val
```

Time: O(n) | Space: O(n)

---

## 22. Sum of Left Leaves (BFS)

Sum all left leaf values. A left leaf is a leaf that is the left child of its parent. BFS. When enqueueing, pass a flag indicating if the node is a left child. Sum when we see a leaf that is a left child.

```python
from collections import deque
def sumOfLeftLeaves(root):
    if not root:
        return 0
    q = deque([(root, False)])
    total = 0
    while q:
        node, is_left = q.popleft()
        if is_left and not node.left and not node.right:
            total += node.val
        if node.left:
            q.append((node.left, True))
        if node.right:
            q.append((node.right, False))
    return total
```

Time: O(n) | Space: O(n)

---

## 23. Path Sum (BFS)

Check if there exists a root-to-leaf path with given sum. BFS with (node, remaining_sum). When we reach a leaf, check if remaining_sum equals node value.

```python
from collections import deque
def hasPathSum(root, targetSum):
    if not root:
        return False
    q = deque([(root, targetSum - root.val)])
    while q:
        node, rem = q.popleft()
        if not node.left and not node.right and rem == 0:
            return True
        if node.left:
            q.append((node.left, rem - node.left.val))
        if node.right:
            q.append((node.right, rem - node.right.val))
    return False
```

Time: O(n) | Space: O(n)

---

## 24. Cousins in Binary Tree

Two nodes are cousins if same depth but different parents. Check if x and y are cousins. BFS. Track parent and depth for each node. When we find x and y, compare their depth and parent.

```python
from collections import deque
def isCousins(root, x, y):
    info = {}
    q = deque([(root, 0, None)])
    while q:
        node, d, parent = q.popleft()
        info[node.val] = (d, parent)
        if node.left:
            q.append((node.left, d + 1, node))
        if node.right:
            q.append((node.right, d + 1, node))
    dx, px = info[x]
    dy, py = info[y]
    return dx == dy and px != py
```

Time: O(n) | Space: O(n)

---

## 25. N-ary Tree Level Order Traversal

Level-order traversal of an N-ary tree (each node has a list of children). Same as binary BFS but enqueue all children from the children list.

```python
from collections import deque
def levelOrder(root):
    if not root:
        return []
    q = deque([root])
    res = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            for child in node.children:
                q.append(child)
        res.append(level)
    return res
```

Time: O(n) | Space: O(n)

---

## 26. Maximum Level Sum of a Binary Tree

Find the level with the maximum sum of node values. Return the smallest level number if tie. BFS. For each level, compute sum. Track max sum and corresponding level.

```python
from collections import deque
def maxLevelSum(root):
    q = deque([root])
    max_sum = float('-inf')
    max_level = level = 0
    while q:
        level += 1
        total = 0
        for _ in range(len(q)):
            node = q.popleft()
            total += node.val
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        if total > max_sum:
            max_sum = total
            max_level = level
    return max_level
```

Time: O(n) | Space: O(n)

---

## 27. Univalued Binary Tree (BFS)

Check if all nodes have the same value. BFS approach. BFS. Compare each node value with root value. Return false on first mismatch.

```python
from collections import deque
def isUnivalTree(root):
    val = root.val
    q = deque([root])
    while q:
        node = q.popleft()
        if node.val != val:
            return False
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return True
```

Time: O(n) | Space: O(n)

---

## 28. Time Needed to Buy Tickets

People in line, each needs tickets[i] tickets. Person at front buys 1 per second. Return seconds for person at index k to finish. Simulate with queue. Each second, front buys 1. If done, leave; else go to back. Count until person k finishes.

```python
from collections import deque
def timeRequiredToBuy(tickets, k):
    q = deque([(i, t) for i, t in enumerate(tickets)])
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

Time: O(sum(tickets)) | Space: O(n)
