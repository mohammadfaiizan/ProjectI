# Hard Linked List Problems

## 1. Merge k Sorted Lists

**Description**: Merge k sorted linked lists into one sorted list.

**Approach**: Min-heap of size k. Push first node of each list. Pop min, append to result, push its next. O(n log k).

---

## 2. Reverse Nodes in k-Group

**Description**: Reverse every k nodes. If remaining < k, leave unchanged.

**Approach**: Iterative: find kth node, reverse sublist, reconnect. Repeat. Or recursive: reverse first k, recurse on rest.

---

## 3. Copy List with Random Pointer

**Description**: Deep copy list with next and random. No extra space for hashmap (O(1) space variant).

**Approach**: Interleaving: insert copy after each original. Set copy.random = original.random.next. Extract copies.

---

## 4. Merge Two Sorted Lists (In-Place, O(1) Space)

**Description**: Merge two sorted lists using only O(1) extra space.

**Approach**: Use one list as base. For each node in other list, find insertion position and insert. No new nodes.

---

## 5. Reverse Linked List (Between m and n, One-Pass)

**Description**: Reverse nodes from position m to n in one pass.

**Approach**: Reach node before m. Repeatedly move curr.next to front of reversed sublist. Do this (n-m) times.

---

## 6. Flatten a Multilevel Doubly Linked List

**Description**: Flatten multilevel list. Each node has next, prev, and child. DFS order.

**Approach**: Recursive flatten. When node has child, flatten child, insert between node and node.next.

---

## 7. LRU Cache

**Description**: Implement LRU cache with get and put in O(1). Evict least recently used when capacity exceeded.

**Approach**: HashMap for O(1) lookup. Doubly linked list for O(1) move-to-front and remove-last. On get: move to front. On put: add to front, evict last if full.

---

## 8. LFU Cache

**Description**: Implement LFU cache. Evict least frequently used. On tie, evict LRU.

**Approach**: HashMap key->node. HashMap freq->doubly linked list of nodes. Track min_freq. On eviction, remove from min_freq list.

---

## 9. All O one Data Structure

**Description**: Implement inc, dec, getMaxKey, getMinKey all in O(1).

**Approach**: HashMap key->count. HashMap count->set of keys. Doubly linked list of counts for max/min. On inc/dec, move key between count buckets.

---

## 10. Design Skiplist

**Description**: Implement skiplist with search, add, erase. Probabilistic multi-level linked structure.

**Approach**: Each node has multiple forward pointers. Levels determined by random. Search: start at top level, go right while next < target, else go down.

---

## 11. Find the Minimum and Maximum Number of Nodes Between Critical Points

**Description**: Critical point: local max or min. Find min and max distance between consecutive critical points.

**Approach**: Traverse, identify critical points (compare with prev and next). Store indices. Min: consecutive difference. Max: first to last.

---

## 12. Reverse Alternating k-Group

**Description**: Reverse first k nodes, skip next k, reverse next k, skip, etc.

**Approach**: For each group: if reverse group, reverse k nodes and connect. If skip group, just advance k nodes. Alternate flag.

---

## 13. Sort a Linked List of 0s 1s 2s

**Description**: Sort list containing only 0, 1, 2. One pass, O(1) space.

**Approach**: Three dummy lists. Traverse once, append each node to correct list. Concatenate 0, 1, 2.

---

## 14. Flatten a Multilevel Linked List (Depth-First)

**Description**: Flatten so child lists come before next sibling. DFS order.

**Approach**: Recursive. Process node, then child (recursive), then next. Build result during recursion.

---

## 15. Clone a Linked List with Next and Random Pointer (O(1) Space)

**Description**: Clone list with random pointer without hashmap.

**Approach**: Interleaving. Create copy after each node. Set copy.random = original.random.next. Split into two lists.

---

## 16. Merge k Sorted Lists (Divide and Conquer)

**Description**: Merge k lists in O(n log k) using divide and conquer.

**Approach**: Pair up lists, merge pairs. Repeat until one list. Each level processes all n nodes. Log k levels.

---

## 17. Reverse a Linked List in Groups of K (Alternating)

**Description**: First k reversed, next k as-is, next k reversed, etc.

**Approach**: Track whether to reverse. For reverse: reverse k nodes. For skip: advance k nodes. Toggle flag.

---

## 18. Flatten a Multilevel Doubly Linked List (Level-Order)

**Description**: Flatten in level order (BFS): all level 0, then all level 1, etc.

**Approach**: Queue. Process node, add next and child to queue. Build result in BFS order.
