# Medium Stack Problems

## 1. Asteroid Collision

Asteroids move at same speed. Positive = right, negative = left. When they meet, smaller explodes. Same size both explode. Return state after all collisions.

**Approach**: Stack. Push positive. For negative: pop while stack has positive and stack[-1] < abs(neg). If equal, pop and skip. If stack empty or top negative, push.

---

## 2. Daily Temperatures

For each day, find number of days until warmer temperature.

**Approach**: Monotonic decreasing stack of indices. Pop when current > stack top; result[popped] = i - popped.

---

## 3. Evaluate Reverse Polish Notation

Evaluate postfix expression with +, -, *, /.

**Approach**: Stack. Operands push. Operator: pop two, apply, push.

---

## 4. Decode String

Decode "3[a2[c]]" to "accaccacc".

**Approach**: Stack. On ']', pop to get substring, pop digits for k, push decoded back.

---

## 5. Remove K Digits

Remove k digits from number string to get smallest number.

**Approach**: Monotonic increasing stack. Pop while top > current and k > 0.

---

## 6. Next Greater Element II

Circular array. Find next greater for each element.

**Approach**: Double array (or modulo). Monotonic stack. Process 2*n indices, only store first n.

---

## 7. Online Stock Span

Stream of prices. For each, return span (consecutive days with price <= today).

**Approach**: Monotonic decreasing stack of (price, span). Pop while top <= current, add spans. Push (price, total_span).

---

## 8. 132 Pattern

Find i < j < k with nums[i] < nums[k] < nums[j].

**Approach**: Traverse right to left. Stack maintains candidates for nums[j]. Track third (nums[k]). When nums[i] < third, found.

---

## 9. Remove Duplicate Letters

Remove duplicates, result lexicographically smallest.

**Approach**: Monotonic stack. Pop while top > current and top appears later. Track last index and seen.

---

## 10. Verify Preorder Serialization of a Binary Tree

Given preorder serialization "9,3,4,#,#,1,#,#,2,#,6,#,#", verify if valid.

**Approach**: Track slot count. Each node consumes 1 slot, adds 2 (for children). '#' consumes 1. Start with 1 slot. Invalid if slots < 0 or non-zero at end.

---

## 11. Exclusive Time of Functions

Given logs with function id, start/end, timestamp. Return exclusive time for each function.

**Approach**: Stack of (id, start_time). On start: push. On end: pop, add (end - start + 1) to id; if stack non-empty, subtract from parent's time.

---

## 12. Flatten Nested List Iterator

Given nested list of integers, implement iterator that flattens it.

**Approach**: Stack stores iterators/lists in reverse order. hasNext: unwind until top is integer. next: return that integer.

---

## 13. Mini Parser

Parse "324" or "[123,[456,[789]]]" into NestedInteger.

**Approach**: Stack. On '[', push new NestedInteger. On ']', pop and add to parent. On digit, parse number and add to top.

---

## 14. Basic Calculator II

String with +, -, *, / and spaces. No parentheses.

**Approach**: Single pass. Track last number and operator. On * or /, apply immediately. On + or -, push number with sign. Sum at end.

---

## 15. Basic Calculator III

String with +, -, *, / and parentheses.

**Approach**: Recursive or stack. On '(', push state (result, sign). On ')', pop and combine. Handle * and / immediately.

---

## 16. Simplify Path

Unix path. Simplify "//foo/../bar/./baz" to "/bar/baz".

**Approach**: Split by '/'. Stack: push non-empty, non-'.'. Pop on '..'. Join.

---

## 17. Longest Absolute File Path

Given file system string, find longest path to a file (containing '.').

**Approach**: Stack of (depth, path_len). Parse level by tabs. Pop while stack depth >= current. Push (depth, len). Update max when path contains '.'.

---

## 18. Ternary Expression Parser

Parse "T?2:3" or "F?1:T?4:5". Return evaluated result.

**Approach**: Parse right to left. Stack. On '?', pop condition, true_val, false_val; push result. On ':', push. On value, push.

---

## 19. Tag Validator

Validate HTML/XML-like tags. Tags must be properly closed and nested.

**Approach**: Stack for tag names. Parse open/close tags. Validate tag names (1-9 chars, uppercase). Match closing with top.

---

## 20. Score of Parentheses

() = 1, AB = A+B, (A) = 2*A.

**Approach**: Stack of scores. On '(', push 0. On ')', pop v, add max(2*v,1) to top.

---

## 21. Minimum Remove to Make Valid Parentheses

Remove minimum parentheses to make valid. Return any valid result.

**Approach**: Stack of indices for '('. On ')', pop or mark remove. Remaining in stack: mark remove. Build excluding those.

---

## 22. Validate Stack Sequences

Given pushed and popped sequences, determine if valid.

**Approach**: Simulate. Push from pushed. When top equals popped[j], pop and j++. Valid if stack empty at end.

---

## 23. Design Browser History

Back, forward, visit(url). Implement with stack (or doubly linked list).

**Approach**: Two stacks: history (back) and forward. Visit: clear forward, push current to history. Back: push current to forward, pop from history. Forward: symmetric.

---

## 24. Design a Stack With Increment Operation

Stack with push, pop, and increment(k, val) which adds val to bottom k elements.

**Approach**: Store (value, increment). On increment(k, val), add to min(k, size) bottom. On pop, add stored increment to result and propagate to next.

---

## 25. Maximum Frequency Stack

Push and pop. Pop returns most frequent element. Tie: most recent.

**Approach**: Map freq. Map freq -> stack of elements. Track max_freq. Push: update freq, add to freq stack. Pop: pop from max_freq stack, update freq, decrement max_freq if empty.

---

## Hard Problems

## 1. Largest Rectangle in Histogram

Find largest rectangle area in histogram.

**Approach**: For each bar, find first smaller left and right. Monotonic increasing stack. Area = height * width.

---

## 2. Maximal Rectangle

Binary matrix. Find largest rectangle of 1s.

**Approach**: Treat each row as histogram base. Heights = consecutive 1s from top. Run largest rectangle per row.

---

## 3. Trapping Rain Water

Compute how much water can be trapped between bars.

**Approach**: Monotonic stack. When popping, water above popped bar = (min(current, stack[-1]) - popped_height) * width.

---

## 4. Basic Calculator

Full calculator with +, -, *, /, parentheses, spaces.

**Approach**: Recursive descent or two stacks. Handle parentheses with recursion/stack of states.

---

## 5. Longest Valid Parentheses

Find length of longest valid parentheses substring.

**Approach**: Stack with -1. On '(', push index. On ')', pop; if empty push index else max_len = max(max_len, i - stack[-1]).

---

## 6. Remove Invalid Parentheses

Remove minimum number of parentheses to make valid. Return all possible results.

**Approach**: BFS. Start with string. Generate all by removing one paren. Check valid. First valid level = answer. Deduplicate.

---

## 7. Number of Atoms

Parse formula like "K4(ON(SO3)2)2" and return count of each atom.

**Approach**: Stack of counts. Parse atoms, numbers, '(', ')'. On ')', multiply by following number, merge into parent.

---

## 8. Decode String (Nested)

"3[a2[c]]" to "accaccacc".

**Approach**: Stack. On ']', pop to '[', get k, push decoded.

---

## 9. Sum of Subarray Minimums

Sum of min over all subarrays. Mod 10^9+7.

**Approach**: For each element as min, count subarrays. Left/right boundaries via monotonic stack. Contribution = arr[i] * (i-left) * (right-i).

---

## 10. Sum of Subarray Ranges

Sum of (max - min) over all subarrays.

**Approach**: Sum of subarray maximums minus sum of subarray minimums. Each uses monotonic stack for boundaries.

---

## 11. Maximum Width Ramp

Find max j-i such that A[i] <= A[j].

**Approach**: Decreasing stack of indices (by value). For each j from right, pop until stack top <= A[j], update width.

---

## 12. Constrained Subsequence Sum

Choose subsequence (no two adjacent) with max sum. At most k apart in original array.

**Approach**: DP with deque for sliding window max. dp[i] = arr[i] + max(0, max(dp[i-k] to dp[i-1])).

---

## 13. Sliding Window Maximum

For each window of size k, return maximum.

**Approach**: Monotonic deque. Front = max. Pop back while back < current. Pop front if out of window.

---

## 14. Minimum Cost Tree From Leaf Values

Given leaf values, build binary tree. Cost of node = product of max leaf in left and right subtree. Minimize sum of costs.

**Approach**: Greedy with stack. Pop while top <= current (current is right max for popped). Cost += popped * min(current, top). Push current.

---

## 15. Count of Smaller Numbers After Self

For each element, count elements to the right that are smaller.

**Approach**: Merge sort with inversion count, or BST, or monotonic stack with binary search. Alternative: process right to left, maintain sorted list, binary search for count.

---

## 16. Max Stack

Stack with push, pop, top, peekMax, popMax. popMax removes and returns the maximum element.

**Approach**: Doubly linked list + treemap (or max heap). Or two stacks: main stack and max stack; popMax requires temporary stack to find and remove.

---

## 17. Expression Add Operators

Given digits and target, insert +, -, * to get target. Return all expressions.

**Approach**: Backtracking. Track current value, previous operand for multiplication. On '*', subtract prev, add prev * current.

---

## 18. Basic Calculator IV

Variables and numbers. Expand and simplify. Return list of terms.

**Approach**: Parse to AST. Recursive descent. Handle +, -, *, parentheses. Simplify by merging like terms. Sort and format output.
