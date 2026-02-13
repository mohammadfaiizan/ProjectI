# Medium Linked List Problems

## 1. Add Two Numbers II

**Description**: Two non-empty lists represent numbers (MSB first). Return sum as a list without reversing.

**Approach**: Use stacks to reverse digit order, then add. Or reverse both lists, add, reverse result.

---

## 2. Swap Nodes in Pairs

**Description**: Swap every two adjacent nodes. Must modify in-place.

**Approach**: Dummy node. For each pair, reverse the two nodes and link to previous group.

---

## 3. Remove Nth Node From End of List

**Description**: Remove nth node from end in one pass.

**Approach**: Lead pointer n+1 steps ahead. When lead reaches null, trailing's next is target.

---

## 4. Rotate List

**Description**: Rotate list to the right by k places.

**Approach**: k = k % n. Find (n-k)th node from start. That node becomes new tail. Its next becomes new head.

---

## 5. Reverse Nodes in k-Group

**Description**: Reverse every k consecutive nodes. Remainder stays as is.

**Approach**: Recursive or iterative. For each group of k, reverse and connect. Handle remainder.

---

## 6. Flatten a Multilevel Doubly Linked List

**Description**: Flatten multilevel list (child pointers) into single-level doubly linked list.

**Approach**: Iterative. When node has child, find tail of current list, append child, set child's prev. Continue.

---

## 7. Copy List with Random Pointer

**Description**: Deep copy list with next and random pointers.

**Approach**: Interleaving method - insert copy after each node, set random, extract copies. Or hashmap.

---

## 8. Reorder List

**Description**: L0 -> Ln -> L1 -> Ln-1 -> L2 -> ...

**Approach**: Find middle (slow-fast), reverse second half, merge by alternating nodes.

---

## 9. Sort List

**Description**: Sort in O(n log n) time, O(1) space (or O(log n) recursion stack).

**Approach**: Merge sort. Split at middle, sort halves, merge.

---

## 10. Insertion Sort List

**Description**: Sort list using insertion sort.

**Approach**: Build sorted list incrementally. For each node, find position in sorted prefix and insert.

---

## 11. Partition List

**Description**: Partition so all nodes < x come before nodes >= x. Preserve relative order.

**Approach**: Two lists (less, ge). Traverse, append to appropriate list. Concatenate.

---

## 12. Remove Duplicates from Sorted List II

**Description**: Delete all nodes that have duplicates. Keep only distinct values.

**Approach**: Dummy. While curr.next exists, skip all nodes equal to curr.next. If no skip happened, add curr to result.

---

## 13. Reverse Linked List II

**Description**: Reverse from position m to n (1-indexed).

**Approach**: Reach node before m. Reverse m to n in one pass (repeatedly move curr.next to front of sublist).

---

## 14. Linked List Random Node

**Description**: Return random node with equal probability. List length unknown.

**Approach**: Reservoir sampling. For each node i, replace result with probability 1/i.

---

## 15. Split Linked List in Parts

**Description**: Split list into k consecutive parts. Lengths should differ by at most 1.

**Approach**: Count n. Base size = n // k, remainder = n % k. First remainder parts get base+1 nodes.

---

## 16. Next Greater Node In Linked List

**Description**: For each node, find next greater value to the right. Store in array.

**Approach**: Convert to array, use monotonic stack to find next greater for each index.

---

## 17. Remove Zero Sum Consecutive Nodes

**Description**: Remove sequences of nodes that sum to zero.

**Approach**: Prefix sum + hashmap. If prefix_sum seen before, remove nodes between. Repeat until no change.

---

## 18. Design Linked List (Doubly)

**Description**: Implement doubly linked list with get, addAtHead, addAtTail, addAtIndex, deleteAtIndex.

**Approach**: Maintain head and tail. Update prev and next for all operations.

---

## 19. Delete the Middle Node of a Linked List

**Description**: Delete the middle node (slow-fast to find, then delete).

**Approach**: Slow-fast to find middle. Need prev of middle to delete. Use dummy or track prev.

---

## 20. Maximum Twin Sum of a Linked List

**Description**: Twin of node i is node n-1-i. Find maximum sum of (node + twin).

**Approach**: Find middle, reverse second half. Traverse first and reversed second in parallel, track max sum.

---

## 21. Delete Nodes and Return Forest

**Description**: Given list and to_delete set, return list of roots of remaining trees (forest).

**Approach**: Track parent. When deleting, add children to result if not in to_delete. Handle root.

---

## 22. Split Circular Linked List

**Description**: Split circular list into two circular lists of roughly equal size.

**Approach**: Slow-fast to find mid. First: head to mid (circular). Second: mid.next to end (circular).

---

## 23. Design Browser History

**Description**: Implement visit, back, forward with max steps.

**Approach**: Doubly linked list. Visit clears forward chain. Back/forward move current pointer.

---

## 24. Merge In Between Linked Lists

**Description**: Remove nodes from list1 between a and b (inclusive), replace with list2.

**Approach**: Find node before a and node after b. Connect before_a to list2 head, list2 tail to after_b.

---

## 25. Swapping Nodes in a Linked List

**Description**: Swap kth node from beginning with kth from end.

**Approach**: Find both nodes (nth from start: traverse k-1; nth from end: lead pointer). Swap their values or links.
