# Easy Queue Problems

## 1. Implement Queue using Stacks

**Description**: Implement a FIFO queue using only two stacks. The queue should support push, pop, peek, and empty operations.

**Approach**: Use one stack for enqueue. For dequeue/peek, transfer all elements to a second stack so the bottom of the first becomes the top of the second. Pop from the second stack for dequeue. Amortized O(1) per operation.

---

## 2. Implement Stack using Queues

**Description**: Implement a LIFO stack using only one or two queues. Support push, pop, top, and empty.

**Approach**: Single queue: push new element, then rotate n-1 elements to bring it to front. Pop is O(1). Two queues: push to q1; for pop, move n-1 from q1 to q2, pop the last from q1, swap queues.

---

## 3. Design Hit Counter

**Description**: Design a hit counter that counts hits in the last 300 seconds. Support hit(timestamp) and getHits(timestamp).

**Approach**: Use a queue to store timestamps. On getHits, remove timestamps older than timestamp - 300, then return queue size.

---

## 4. Number of Recent Calls

**Description**: Design RecentCounter with ping(t). Return the number of requests in the last 3000 ms.

**Approach**: Queue of timestamps. On ping, append t, remove from front while front < t - 3000, return queue size.

---

## 5. Moving Average from Data Stream

**Description**: Calculate the moving average of the last size values. Support next(val).

**Approach**: Queue of size at most `size`. Maintain running sum. When full, dequeue oldest and subtract from sum before enqueueing new.

---

## 6. First Unique Character in a String

**Description**: Find the index of the first non-repeating character in a string. Return -1 if none.

**Approach**: Count frequency of each character. Iterate and return first with count 1. Can also use queue: enqueue chars, dequeue while front has count > 1.

---

## 7. Implement Queue using Stacks (Amortized O(1))

**Description**: Same as problem 1; ensure amortized O(1) per operation.

**Approach**: Stack-in for enqueue, stack-out for dequeue. When stack-out is empty, transfer all from stack-in. Each element is pushed and popped at most twice.

---

## 8. Design Circular Queue

**Description**: Design a circular queue with fixed capacity. Support enQueue, deQueue, Front, Rear, isEmpty, isFull.

**Approach**: Circular array with front and rear indices. Use modulo for wrap-around. Maintain size or sacrifice one slot to distinguish full from empty.

---

## 9. Design Circular Deque

**Description**: Design a double-ended queue with fixed capacity. Support insertFront, insertLast, deleteFront, deleteLast, getFront, getRear, isEmpty, isFull.

**Approach**: Circular array. Front and rear can grow in both directions. Use modulo arithmetic for indices.

---

## 10. Generate Binary Numbers from 1 to N

**Description**: Generate first n binary numbers: "1", "10", "11", "100", etc.

**Approach**: BFS with queue. Start with "1". Dequeue, output, enqueue current+"0" and current+"1".

---

## 11. Reverse First K Elements of Queue

**Description**: Given a queue and integer k, reverse the order of the first k elements.

**Approach**: Push first k elements to a stack. Pop back to queue. Rotate remaining n-k elements to the back.

---

## 12. Level Order Traversal of Binary Tree

**Description**: Return level-order traversal (BFS) of a binary tree.

**Approach**: Queue. Enqueue root. While queue not empty: dequeue, process, enqueue left and right children.

---

## 13. Average of Levels in Binary Tree

**Description**: Return the average value of nodes at each level.

**Approach**: BFS with level tracking. For each level, sum all values and divide by count.

---

## 14. Minimum Depth of Binary Tree

**Description**: Find the minimum depth (shortest path from root to leaf).

**Approach**: BFS. Return depth when we first encounter a node with no children.

---

## 15. Symmetric Tree (BFS variant)

**Description**: Check if binary tree is mirror of itself. Can use BFS comparing level by level.

**Approach**: BFS, store each level. Check if each level is palindrome. Or use two queues for left and right subtrees.

---

## 16. Merge Two Binary Trees (BFS)

**Description**: Merge two binary trees by summing overlapping nodes. Can implement with BFS.

**Approach**: BFS both trees in parallel. When both have nodes at a position, sum values. When one is null, use the other.

---

## 17. Invert Binary Tree (BFS)

**Description**: Invert a binary tree (swap left and right children). BFS implementation.

**Approach**: BFS. For each node, swap its left and right children before enqueueing them.

---

## 18. Same Tree (BFS)

**Description**: Check if two binary trees are identical. BFS comparison.

**Approach**: BFS both trees in lockstep. Compare values at each step. Structure must match.

---

## 19. Maximum Depth of Binary Tree (BFS)

**Description**: Find the maximum depth of a binary tree. BFS variant.

**Approach**: BFS with level counter. Increment level after processing each level. Return final level.

---

## 20. Binary Tree Level Order Traversal II

**Description**: Level-order traversal but return levels from bottom to top.

**Approach**: Standard BFS level-order, then reverse the list of levels.

---

## 21. Find Bottom Left Tree Value

**Description**: Find the value of the leftmost node in the last row of the tree.

**Approach**: BFS level-order. Return the first node value of the last level.

---

## 22. Sum of Left Leaves (BFS)

**Description**: Sum all left leaf values. A left leaf is a leaf that is the left child of its parent.

**Approach**: BFS. When enqueueing, pass a flag indicating if the node is a left child. Sum when we see a leaf that is a left child.

---

## 23. Path Sum (BFS)

**Description**: Check if there exists a root-to-leaf path with given sum.

**Approach**: BFS with (node, remaining_sum). When we reach a leaf, check if remaining_sum equals node value.

---

## 24. Cousins in Binary Tree

**Description**: Two nodes are cousins if same depth but different parents. Check if x and y are cousins.

**Approach**: BFS. Track parent and depth for each node. When we find x and y, compare their depth and parent.

---

## 25. N-ary Tree Level Order Traversal

**Description**: Level-order traversal of an N-ary tree (each node has a list of children).

**Approach**: Same as binary BFS but enqueue all children from the children list.

---

## 26. Maximum Level Sum of a Binary Tree

**Description**: Find the level with the maximum sum of node values. Return the smallest level number if tie.

**Approach**: BFS. For each level, compute sum. Track max sum and corresponding level.

---

## 27. Univalued Binary Tree (BFS)

**Description**: Check if all nodes have the same value. BFS approach.

**Approach**: BFS. Compare each node value with root value. Return false on first mismatch.

---

## 28. Time Needed to Buy Tickets

**Description**: People in line, each needs tickets[i] tickets. Person at front buys 1 per second. Return seconds for person at index k to finish.

**Approach**: Simulate with queue. Each second, front buys 1. If done, leave; else go to back. Count until person k finishes.
