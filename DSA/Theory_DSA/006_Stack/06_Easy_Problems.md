# Easy Stack Problems

## 1. Implement Stack Using Queues

Implement a last-in-first-out (LIFO) stack using only two queues. Support push, pop, top, and empty.

**Approach**: Use two queues. On push, add to q1. On pop, move n-1 elements from q1 to q2, dequeue the last from q1, swap queues. Alternatively, single queue: push new element then rotate n-1 elements to bring it to front.

---

## 2. Valid Parentheses

Given a string s containing '(', ')', '{', '}', '[', ']', determine if the input is valid. Open brackets must be closed by the same type and in correct order.

**Approach**: Stack. Push opening brackets. On closing, pop and check match. Valid if stack empty at end.

---

## 3. Min Stack

Design a stack that supports push, pop, top, and retrieving the minimum element in O(1) time.

**Approach**: Store (value, min_so_far) in each stack element. On push, min = min(new_val, current_min). getMin returns top's min.

---

## 4. Backspace String Compare

Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means backspace.

**Approach**: Simulate typing with stack. Push char on letter, pop on '#'. Compare resulting strings.

---

## 5. Remove Outermost Parentheses

Remove the outermost parentheses of every primitive string in the valid parentheses string S.

**Approach**: Track depth. Skip char when depth goes 0->1 (opening) or 1->0 (closing). Otherwise append.

---

## 6. Make The String Great

Given a string s of lower and upper case letters, repeatedly remove adjacent pairs of same letter (one lower, one upper) until no more can be removed.

**Approach**: Stack. Push char. If top is same letter opposite case, pop. Return ''.join(stack).

---

## 7. Baseball Game

You are keeping score for a baseball game. Operations: integer (add to record), '+' (sum of previous two), 'D' (double previous), 'C' (remove previous). Return sum of record.

**Approach**: Stack. On integer: push. On '+': push(sum of top two). On 'D': push(2*top). On 'C': pop. Return sum(stack).

---

## 8. Remove All Adjacent Duplicates In String

Given a string, repeatedly remove adjacent duplicate characters until no more can be removed.

**Approach**: Stack. Push char. If top equals current, pop. Return ''.join(stack).

---

## 9. Final Prices With a Special Discount

For each item, discount = first smaller price to the right. Final price = price - discount.

**Approach**: Monotonic increasing stack. For each index, pop while top > current; for popped, discount = current. Store result.

---

## 10. Crawler Log Folder

Operations: "../" (go up), "./" (stay), "x/" (go into x). Start at main. Return minimum operations to go back to main.

**Approach**: Stack for depth. "../" pop if non-empty. "x/" push. Return len(stack).

---

## 11. Maximum Nesting Depth of Parentheses

Return maximum nesting depth of valid parentheses string.

**Approach**: Track depth. On '(', depth += 1, update max. On ')', depth -= 1.

---

## 12. Reverse String

Reverse a string in-place (or return reversed string).

**Approach**: Stack: push all, pop all. Or two pointers swap.

---

## 13. Implement Queue Using Stacks

Implement FIFO queue using only two stacks.

**Approach**: Two stacks (input, output). Push to input. Pop/peek: if output empty, transfer all from input to output, then pop/peek from output.

---

## 14. Evaluate Reverse Polish Notation

Evaluate expression in postfix notation. Valid operators: +, -, *, /.

**Approach**: Stack. Operands: push. Operator: pop two, apply, push. Return top.

---

## 15. Decode String

Decode string like "3[a2[c]]" to "accaccacc".

**Approach**: Stack. Push until ']'. Pop to get substring, pop digits for k, push decoded string back.

---

## 16. Remove K Digits

Given a non-negative integer num and k, remove k digits to get the smallest possible number.

**Approach**: Monotonic increasing stack. Pop while top > current and k > 0. Handle leading zeros.

---

## 17. Next Greater Element I

Given nums1 (subset of nums2), for each element in nums1 find next greater in nums2.

**Approach**: Build next greater map for nums2 using monotonic stack. Lookup for nums1.

---

## 18. Daily Temperatures

For each day, return number of days to wait until a warmer temperature.

**Approach**: Monotonic decreasing stack of indices. Pop when current > stack top; result[popped] = i - popped.

---

## 19. Score of Parentheses

() has score 1. AB has score A+B. (A) has score 2*A. Return total score.

**Approach**: Stack of scores. On '(', push 0. On ')', pop v, add max(2*v,1) to new top.

---

## 20. Minimum Add to Make Parentheses Valid

Return minimum number of '(' or ')' to add to make string valid.

**Approach**: Track open_needed and close_needed. On '(', close_needed += 1. On ')', if close_needed > 0 decrement else open_needed += 1.

---

## 21. Valid Parentheses (Multiple Types)

Check if string with '()', '[]', '{}' is valid.

**Approach**: Stack with matching map. Push opening. On closing, pop and verify match.

---

## 22. Remove Duplicate Letters

Given a string, remove duplicate letters so every letter appears once and result is lexicographically smallest.

**Approach**: Monotonic stack. Pop while top > current and top appears later. Track last index and seen.

---

## 23. Baseball Game (Alternate)

Same as problem 7. Stack-based simulation.

---

## 24. Simplify Path

Given an absolute path for a Unix file system, return simplified canonical path.

**Approach**: Split by '/'. Stack: push non-empty non-'.' segments. Pop on '..'. Join with '/'.

---

## 25. Min Remove to Make Valid Parentheses

Remove minimum number of parentheses to make string valid. Return any valid result.

**Approach**: Stack of indices for '('. On ')', pop or mark for removal. Remaining in stack: mark for removal. Build result excluding those indices.
