# Easy Linked List Problems

## 1. Reverse Linked List

**Description**: Reverse a singly linked list.

**Approach**: Iterative three-pointer (prev, curr, next) or recursive. Change each node's next to point to previous.

---

## 2. Merge Two Sorted Lists

**Description**: Merge two sorted linked lists into one sorted list.

**Approach**: Two pointers, compare and link smaller node. Append remainder when one list exhausts.

---

## 3. Linked List Cycle

**Description**: Determine if a linked list has a cycle.

**Approach**: Floyd's tortoise and hare. Slow and fast pointers; if they meet, cycle exists.

---

## 4. Remove Duplicates from Sorted List

**Description**: Delete all duplicates such that each element appears only once.

**Approach**: Single pass. If curr.data == curr.next.data, skip curr.next. Otherwise advance.

---

## 5. Remove Linked List Elements

**Description**: Remove all nodes with a given value.

**Approach**: Handle head deletions first with a loop. Then traverse and skip nodes with target value.

---

## 6. Palindrome Linked List

**Description**: Check if a linked list is a palindrome.

**Approach**: Find middle with slow-fast, reverse second half, compare both halves. Restore second half.

---

## 7. Middle of the Linked List

**Description**: Return the middle node. If two middles, return the second.

**Approach**: Slow-fast pointers. When fast reaches end, slow is at middle.

---

## 8. Delete Node in a Linked List

**Description**: Delete a node given only a pointer to that node (not the head).

**Approach**: Copy next node's data to current, then delete next node.

---

## 9. Remove Nth Node From End of List

**Description**: Remove the nth node from the end in one pass.

**Approach**: Dummy node. Lead pointer n+1 steps ahead. When lead reaches end, trailing pointer's next is the node to remove.

---

## 10. Convert Binary Number in a Linked List to Integer

**Description**: Each node holds 0 or 1. Return decimal value of the binary number.

**Approach**: Traverse and accumulate: result = result * 2 + node.val.

---

## 11. Design Linked List

**Description**: Implement get, addAtHead, addAtTail, addAtIndex, deleteAtIndex.

**Approach**: Maintain head and optionally tail. Handle edge cases for index 0 and out of bounds.

---

## 12. Merge Two Sorted Lists (In-Place)

**Description**: Merge two sorted lists without creating new nodes.

**Approach**: Use one list as base, insert nodes from the other in correct position. Or use dummy and relink.

---

## 13. Intersection of Two Linked Lists

**Description**: Find the node where two lists intersect (by reference). Return null if no intersection.

**Approach**: Find lengths, advance longer list by difference, traverse both in parallel until same node.

---

## 14. Reverse Linked List II

**Description**: Reverse nodes from position m to n (1-indexed).

**Approach**: Reach node before m, reverse m to n using one-pass reversal, reconnect.

---

## 15. Swap Nodes in Pairs

**Description**: Swap every two adjacent nodes.

**Approach**: Dummy node. For each pair, swap first and second. Advance by two.

---

## 16. Add Two Numbers

**Description**: Two lists represent numbers in reverse (LSB first). Return sum as a list.

**Approach**: Add digit by digit, maintain carry. Create new nodes for result.

---

## 17. Remove Duplicates from Sorted List II

**Description**: Remove all nodes that have duplicate values (keep only distinct values).

**Approach**: Dummy node. Skip all nodes with same value as next. If no duplicate, add to result.

---

## 18. Partition List

**Description**: Partition list so all nodes < x come before nodes >= x. Preserve order.

**Approach**: Two dummy lists (less, ge). Traverse and append to appropriate list. Concatenate.

---

## 19. Odd Even Linked List

**Description**: Group odd-indexed nodes first, then even-indexed. Use O(1) space.

**Approach**: Two pointers for odd and even lists. Interleave in one pass.

---

## 20. Sort List

**Description**: Sort linked list in O(n log n) time and O(1) space (excluding recursion).

**Approach**: Merge sort. Find middle with slow-fast, split, recursively sort, merge.

---

## 21. Reorder List

**Description**: L0 -> Ln -> L1 -> Ln-1 -> L2 -> ...

**Approach**: Find middle, reverse second half, interleave first and reversed second.

---

## 22. Copy List with Random Pointer

**Description**: Deep copy a list where each node has next and random pointer.

**Approach**: HashMap: old node -> new node. Two passes: create nodes, set next and random.

---

## 23. Linked List Cycle II

**Description**: Return the node where the cycle begins. Return null if no cycle.

**Approach**: Floyd's cycle detection. After meeting, one pointer at head, both move one step. Meeting point is cycle start.

---

## 24. Flatten a Multilevel Doubly Linked List

**Description**: Flatten a multilevel doubly linked list (nodes can have child pointers).

**Approach**: DFS or iterative. When node has child, append child list to current tail, continue.

---

## 25. Design Browser History

**Description**: Implement back, forward, visit for browser history using doubly linked list.

**Approach**: Doubly linked list with current pointer. Visit clears forward history.
