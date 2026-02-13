# Easy Stack Problems

## 1. Implement Stack Using Queues

Implement a last-in-first-out (LIFO) stack using only two queues. Support push, pop, top, and empty. Use two queues. On push, add to q1. On pop, move n-1 elements from q1 to q2, dequeue the last from q1, swap queues. Alternatively, single queue: push new element then rotate n-1 elements to bring it to front.

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

Time: O(1) push amortized, O(1) pop/top | Space: O(n)

---

## 2. Valid Parentheses

Given a string s containing '(', ')', '{', '}', '[', ']', determine if the input is valid. Open brackets must be closed by the same type and in correct order. Stack. Push opening brackets. On closing, pop and check match. Valid if stack empty at end.

```python
def isValid(s):
    st = []
    m = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in '({[':
            st.append(c)
        elif not st or st[-1] != m[c]:
            return False
        else:
            st.pop()
    return len(st) == 0
```

Time: O(n) | Space: O(n)

---

## 3. Min Stack

Design a stack that supports push, pop, top, and retrieving the minimum element in O(1) time. Store (value, min_so_far) in each stack element. On push, min = min(new_val, current_min). getMin returns top's min.

```python
class MinStack:
    def __init__(self):
        self.st = []

    def push(self, val):
        m = min(val, self.st[-1][1]) if self.st else val
        self.st.append((val, m))

    def pop(self):
        self.st.pop()

    def top(self):
        return self.st[-1][0]

    def getMin(self):
        return self.st[-1][1]
```

Time: O(1) all ops | Space: O(n)

---

## 4. Backspace String Compare

Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means backspace. Simulate typing with stack. Push char on letter, pop on '#'. Compare resulting strings.

```python
def backspaceCompare(s, t):
    def build(s):
        st = []
        for c in s:
            if c == '#':
                if st:
                    st.pop()
            else:
                st.append(c)
        return ''.join(st)
    return build(s) == build(t)
```

Time: O(n+m) | Space: O(n+m)

---

## 5. Remove Outermost Parentheses

Remove the outermost parentheses of every primitive string in the valid parentheses string S. Track depth. Skip char when depth goes 0->1 (opening) or 1->0 (closing). Otherwise append.

```python
def removeOuterParentheses(s):
    res = []
    depth = 0
    for c in s:
        if c == '(':
            depth += 1
            if depth > 1:
                res.append(c)
        else:
            depth -= 1
            if depth > 0:
                res.append(c)
    return ''.join(res)
```

Time: O(n) | Space: O(n)

---

## 6. Make The String Great

Given a string s of lower and upper case letters, repeatedly remove adjacent pairs of same letter (one lower, one upper) until no more can be removed. Stack. Push char. If top is same letter opposite case, pop. Return ''.join(stack).

```python
def makeGood(s):
    st = []
    for c in s:
        if st and st[-1] != c and st[-1].lower() == c.lower():
            st.pop()
        else:
            st.append(c)
    return ''.join(st)
```

Time: O(n) | Space: O(n)

---

## 7. Baseball Game

You are keeping score for a baseball game. Operations: integer (add to record), '+' (sum of previous two), 'D' (double previous), 'C' (remove previous). Return sum of record. Stack. On integer: push. On '+': push(sum of top two). On 'D': push(2*top). On 'C': pop. Return sum(stack).

```python
def calPoints(ops):
    st = []
    for op in ops:
        if op == '+':
            st.append(st[-1] + st[-2])
        elif op == 'D':
            st.append(st[-1] * 2)
        elif op == 'C':
            st.pop()
        else:
            st.append(int(op))
    return sum(st)
```

Time: O(n) | Space: O(n)

---

## 8. Remove All Adjacent Duplicates In String

Given a string, repeatedly remove adjacent duplicate characters until no more can be removed. Stack. Push char. If top equals current, pop. Return ''.join(stack).

```python
def removeDuplicates(s):
    st = []
    for c in s:
        if st and st[-1] == c:
            st.pop()
        else:
            st.append(c)
    return ''.join(st)
```

Time: O(n) | Space: O(n)

---

## 9. Final Prices With a Special Discount

For each item, discount = first smaller price to the right. Final price = price - discount. Monotonic increasing stack. For each index, pop while top > current; for popped, discount = current. Store result.

```python
def finalPrices(prices):
    st = []
    res = prices[:]
    for i in range(len(prices)):
        while st and prices[st[-1]] >= prices[i]:
            res[st.pop()] -= prices[i]
        st.append(i)
    return res
```

Time: O(n) | Space: O(n)

---

## 10. Crawler Log Folder

Operations: "../" (go up), "./" (stay), "x/" (go into x). Start at main. Return minimum operations to go back to main. Stack for depth. "../" pop if non-empty. "x/" push. Return len(stack).

```python
def minOperations(logs):
    depth = 0
    for log in logs:
        if log == '../':
            depth = max(0, depth - 1)
        elif log != './':
            depth += 1
    return depth
```

Time: O(n) | Space: O(1)

---

## 11. Maximum Nesting Depth of Parentheses

Return maximum nesting depth of valid parentheses string. Track depth. On '(', depth += 1, update max. On ')', depth -= 1.

```python
def maxDepth(s):
    depth = max_d = 0
    for c in s:
        if c == '(':
            depth += 1
            max_d = max(max_d, depth)
        elif c == ')':
            depth -= 1
    return max_d
```

Time: O(n) | Space: O(1)

---

## 12. Reverse String

Reverse a string in-place (or return reversed string). Stack: push all, pop all. Or two pointers swap.

```python
def reverseString(s):
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
```

Time: O(n) | Space: O(1)

---

## 13. Implement Queue Using Stacks

Implement FIFO queue using only two stacks. Two stacks (input, output). Push to input. Pop/peek: if output empty, transfer all from input to output, then pop/peek from output.

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

## 14. Evaluate Reverse Polish Notation

Evaluate expression in postfix notation. Valid operators: +, -, *, /. Stack. Operands: push. Operator: pop two, apply, push. Return top.

```python
def evalRPN(tokens):
    st = []
    for t in tokens:
        if t in '+-*/':
            b, a = st.pop(), st.pop()
            st.append(int(eval(f'{a}{t}{b}')))
        else:
            st.append(int(t))
    return st[-1]
```

Time: O(n) | Space: O(n)

---

## 15. Decode String

Decode string like "3[a2[c]]" to "accaccacc". Stack. Push until ']'. Pop to get substring, pop digits for k, push decoded string back.

```python
def decodeString(s):
    st = []
    for c in s:
        if c == ']':
            sub = ''
            while st[-1] != '[':
                sub = st.pop() + sub
            st.pop()
            k = ''
            while st and st[-1].isdigit():
                k = st.pop() + k
            st.append(int(k) * sub)
        else:
            st.append(c)
    return ''.join(st)
```

Time: O(n) | Space: O(n)

---

## 16. Remove K Digits

Given a non-negative integer num and k, remove k digits to get the smallest possible number. Monotonic increasing stack. Pop while top > current and k > 0. Handle leading zeros.

```python
def removeKdigits(num, k):
    st = []
    for d in num:
        while k and st and st[-1] > d:
            st.pop()
            k -= 1
        st.append(d)
    while k:
        st.pop()
        k -= 1
    res = ''.join(st).lstrip('0')
    return res or '0'
```

Time: O(n) | Space: O(n)

---

## 17. Next Greater Element I

Given nums1 (subset of nums2), for each element in nums1 find next greater in nums2. Build next greater map for nums2 using monotonic stack. Lookup for nums1.

```python
def nextGreaterElement(nums1, nums2):
    m = {}
    st = []
    for x in nums2:
        while st and st[-1] < x:
            m[st.pop()] = x
        st.append(x)
    return [m.get(x, -1) for x in nums1]
```

Time: O(n+m) | Space: O(n)

---

## 18. Daily Temperatures

For each day, return number of days to wait until a warmer temperature. Monotonic decreasing stack of indices. Pop when current > stack top; result[popped] = i - popped.

```python
def dailyTemperatures(temperatures):
    st = []
    res = [0] * len(temperatures)
    for i in range(len(temperatures)):
        while st and temperatures[st[-1]] < temperatures[i]:
            j = st.pop()
            res[j] = i - j
        st.append(i)
    return res
```

Time: O(n) | Space: O(n)

---

## 19. Score of Parentheses

() has score 1. AB has score A+B. (A) has score 2*A. Return total score. Stack of scores. On '(', push 0. On ')', pop v, add max(2*v,1) to new top.

```python
def scoreOfParentheses(s):
    st = [0]
    for c in s:
        if c == '(':
            st.append(0)
        else:
            v = st.pop()
            st[-1] += max(2 * v, 1)
    return st[0]
```

Time: O(n) | Space: O(n)

---

## 20. Minimum Add to Make Parentheses Valid

Return minimum number of '(' or ')' to add to make string valid. Track open_needed and close_needed. On '(', close_needed += 1. On ')', if close_needed > 0 decrement else open_needed += 1.

```python
def minAddToMakeValid(s):
    open_needed = close_needed = 0
    for c in s:
        if c == '(':
            close_needed += 1
        else:
            if close_needed:
                close_needed -= 1
            else:
                open_needed += 1
    return open_needed + close_needed
```

Time: O(n) | Space: O(1)

---

## 21. Valid Parentheses (Multiple Types)

Check if string with '()', '[]', '{}' is valid. Stack with matching map. Push opening. On closing, pop and verify match.

```python
def isValid(s):
    st = []
    m = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in '({[':
            st.append(c)
        elif not st or st[-1] != m[c]:
            return False
        else:
            st.pop()
    return len(st) == 0
```

Time: O(n) | Space: O(n)

---

## 22. Remove Duplicate Letters

Given a string, remove duplicate letters so every letter appears once and result is lexicographically smallest. Monotonic stack. Pop while top > current and top appears later. Track last index and seen.

```python
def removeDuplicateLetters(s):
    last = {c: i for i, c in enumerate(s)}
    st = []
    seen = set()
    for i, c in enumerate(s):
        if c in seen:
            continue
        while st and st[-1] > c and last[st[-1]] > i:
            seen.discard(st.pop())
        st.append(c)
        seen.add(c)
    return ''.join(st)
```

Time: O(n) | Space: O(n)

---

## 23. Baseball Game (Alternate)

Same as problem 7. Stack-based simulation.

```python
def calPoints(ops):
    st = []
    for op in ops:
        if op == '+':
            st.append(st[-1] + st[-2])
        elif op == 'D':
            st.append(st[-1] * 2)
        elif op == 'C':
            st.pop()
        else:
            st.append(int(op))
    return sum(st)
```

Time: O(n) | Space: O(n)

---

## 24. Simplify Path

Given an absolute path for a Unix file system, return simplified canonical path. Split by '/'. Stack: push non-empty non-'.' segments. Pop on '..'. Join with '/'.

```python
def simplifyPath(path):
    parts = path.split('/')
    st = []
    for p in parts:
        if p in ('', '.'):
            continue
        if p == '..':
            if st:
                st.pop()
        else:
            st.append(p)
    return '/' + '/'.join(st)
```

Time: O(n) | Space: O(n)

---

## 25. Min Remove to Make Valid Parentheses

Remove minimum number of parentheses to make string valid. Return any valid result. Stack of indices for '('. On ')', pop or mark for removal. Remaining in stack: mark for removal. Build result excluding those indices.

```python
def minRemoveToMakeValid(s):
    st = []
    remove = set()
    for i, c in enumerate(s):
        if c == '(':
            st.append(i)
        elif c == ')':
            if st:
                st.pop()
            else:
                remove.add(i)
    remove.update(st)
    return ''.join(c for i, c in enumerate(s) if i not in remove)
```

Time: O(n) | Space: O(n)
