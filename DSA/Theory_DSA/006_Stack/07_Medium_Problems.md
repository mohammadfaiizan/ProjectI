# Medium Stack Problems

## 1. Asteroid Collision

Asteroids move at same speed. Positive = right, negative = left. When they meet, smaller explodes. Same size both explode. Return state after all collisions. Stack. Push positive. For negative: pop while stack has positive and stack[-1] < abs(neg). If equal, pop and skip. If stack empty or top negative, push.

```python
def asteroidCollision(asteroids):
    st = []
    for a in asteroids:
        while st and st[-1] > 0 and a < 0:
            if st[-1] < -a:
                st.pop()
            elif st[-1] == -a:
                st.pop()
                a = 0
                break
            else:
                a = 0
                break
        if a:
            st.append(a)
    return st
```

Time: O(n) | Space: O(n)

---

## 2. Daily Temperatures

For each day, find number of days until warmer temperature. Monotonic decreasing stack of indices. Pop when current > stack top; result[popped] = i - popped.

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

## 3. Evaluate Reverse Polish Notation

Evaluate postfix expression with +, -, *, /. Stack. Operands push. Operator: pop two, apply, push.

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

## 4. Decode String

Decode "3[a2[c]]" to "accaccacc". Stack. On ']', pop to get substring, pop digits for k, push decoded back.

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

## 5. Remove K Digits

Remove k digits from number string to get smallest number. Monotonic increasing stack. Pop while top > current and k > 0.

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

## 6. Next Greater Element II

Circular array. Find next greater for each element. Double array (or modulo). Monotonic stack. Process 2*n indices, only store first n.

```python
def nextGreaterElements(nums):
    n = len(nums)
    res = [-1] * n
    st = []
    for i in range(2 * n):
        idx = i % n
        while st and nums[st[-1]] < nums[idx]:
            res[st.pop()] = nums[idx]
        if i < n:
            st.append(idx)
    return res
```

Time: O(n) | Space: O(n)

---

## 7. Online Stock Span

Stream of prices. For each, return span (consecutive days with price <= today). Monotonic decreasing stack of (price, span). Pop while top <= current, add spans. Push (price, total_span).

```python
class StockSpanner:
    def __init__(self):
        self.st = []

    def next(self, price):
        span = 1
        while self.st and self.st[-1][0] <= price:
            span += self.st.pop()[1]
        self.st.append((price, span))
        return span
```

Time: O(1) amortized | Space: O(n)

---

## 8. 132 Pattern

Find i < j < k with nums[i] < nums[k] < nums[j]. Traverse right to left. Stack maintains candidates for nums[j]. Track third (nums[k]). When nums[i] < third, found.

```python
def find132pattern(nums):
    third = float('-inf')
    st = []
    for i in range(len(nums) - 1, -1, -1):
        if nums[i] < third:
            return True
        while st and st[-1] < nums[i]:
            third = st.pop()
        st.append(nums[i])
    return False
```

Time: O(n) | Space: O(n)

---

## 9. Remove Duplicate Letters

Remove duplicates, result lexicographically smallest. Monotonic stack. Pop while top > current and top appears later. Track last index and seen.

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

## 10. Verify Preorder Serialization of a Binary Tree

Given preorder serialization "9,3,4,#,#,1,#,#,2,#,6,#,#", verify if valid. Track slot count. Each node consumes 1 slot, adds 2 (for children). '#' consumes 1. Start with 1 slot. Invalid if slots < 0 or non-zero at end.

```python
def isValidSerialization(preorder):
    slots = 1
    for node in preorder.split(','):
        slots -= 1
        if slots < 0:
            return False
        if node != '#':
            slots += 2
    return slots == 0
```

Time: O(n) | Space: O(1)

---

## 11. Exclusive Time of Functions

Given logs with function id, start/end, timestamp. Return exclusive time for each function. Stack of (id, start_time). On start: push. On end: pop, add (end - start + 1) to id; if stack non-empty, subtract from parent's time.

```python
def exclusiveTime(n, logs):
    res = [0] * n
    st = []
    for log in logs:
        fid, typ, ts = log.split(':')
        fid, ts = int(fid), int(ts)
        if typ == 'start':
            if st:
                res[st[-1][0]] += ts - st[-1][1]
            st.append([fid, ts])
        else:
            res[fid] += ts - st[-1][1] + 1
            st.pop()
            if st:
                st[-1][1] = ts + 1
    return res
```

Time: O(n) | Space: O(n)

---

## 12. Flatten Nested List Iterator

Given nested list of integers, implement iterator that flattens it. Stack stores iterators/lists in reverse order. hasNext: unwind until top is integer. next: return that integer.

```python
class NestedIterator:
    def __init__(self, nestedList):
        self.st = [iter(nestedList)]
        self.cur = None

    def next(self):
        return self.cur

    def hasNext(self):
        while self.st:
            try:
                x = next(self.st[-1])
                if x.isInteger():
                    self.cur = x.getInteger()
                    return True
                self.st.append(iter(x.getList()))
            except StopIteration:
                self.st.pop()
        return False
```

Time: O(1) amortized | Space: O(depth)

---

## 13. Mini Parser

Parse "324" or "[123,[456,[789]]]" into NestedInteger. Stack. On '[', push new NestedInteger. On ']', pop and add to parent. On digit, parse number and add to top.

```python
def deserialize(s):
    if s[0] != '[':
        return NestedInteger(int(s))
    st = []
    i = 0
    while i < len(s):
        if s[i] == '[':
            st.append(NestedInteger())
            i += 1
        elif s[i] == ']':
            top = st.pop()
            if st:
                st[-1].add(top)
            else:
                return top
            i += 1
        elif s[i] == ',':
            i += 1
        else:
            j = i
            while j < len(s) and s[j] in '-0123456789':
                j += 1
            st[-1].add(NestedInteger(int(s[i:j])))
            i = j
    return st[-1]
```

Time: O(n) | Space: O(n)

---

## 14. Basic Calculator II

String with +, -, *, / and spaces. No parentheses. Single pass. Track last number and operator. On * or /, apply immediately. On + or -, push number with sign. Sum at end.

```python
def calculate(s):
    num, op = 0, '+'
    st = []
    for i, c in enumerate(s + '+'):
        if c.isdigit():
            num = num * 10 + int(c)
        elif c in '+-*/':
            if op == '+':
                st.append(num)
            elif op == '-':
                st.append(-num)
            elif op == '*':
                st.append(st.pop() * num)
            elif op == '/':
                st.append(int(st.pop() / num))
            op = c
            num = 0
    return sum(st)
```

Time: O(n) | Space: O(n)

---

## 15. Basic Calculator III

String with +, -, *, / and parentheses. Recursive or stack. On '(', push state (result, sign). On ')', pop and combine. Handle * and / immediately.

```python
def calculate(s):
    def calc(i):
        st, num, op = [], 0, '+'
        while i < len(s):
            c = s[i]
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == '(':
                num, i = calc(i + 1)
            elif c in '+-*/':
                if op == '+': st.append(num)
                elif op == '-': st.append(-num)
                elif op == '*': st.append(st.pop() * num)
                else: st.append(int(st.pop() / num))
                op, num = c, 0
            elif c == ')':
                if op == '+': st.append(num)
                elif op == '-': st.append(-num)
                elif op == '*': st.append(st.pop() * num)
                else: st.append(int(st.pop() / num))
                return sum(st), i
            i += 1
        if op == '+': st.append(num)
        elif op == '-': st.append(-num)
        elif op == '*': st.append(st.pop() * num)
        else: st.append(int(st.pop() / num))
        return sum(st), i
    return calc(0)[0]
```

Time: O(n) | Space: O(n)

---

## 16. Simplify Path

Unix path. Simplify "//foo/../bar/./baz" to "/bar/baz". Split by '/'. Stack: push non-empty, non-'.'. Pop on '..'. Join.

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

## 17. Longest Absolute File Path

Given file system string, find longest path to a file (containing '.'). Stack of (depth, path_len). Parse level by tabs. Pop while stack depth >= current. Push (depth, len). Update max when path contains '.'.

```python
def lengthLongestPath(input):
    st = [0]
    max_len = 0
    for line in input.split('\n'):
        depth = line.count('\t')
        name = line.replace('\t', '')
        while len(st) > depth + 1:
            st.pop()
        length = st[-1] + len(name) + (1 if depth else 0)
        if '.' in name:
            max_len = max(max_len, length)
        else:
            st.append(length)
    return max_len
```

Time: O(n) | Space: O(depth)

---

## 18. Ternary Expression Parser

Parse "T?2:3" or "F?1:T?4:5". Return evaluated result. Parse right to left. Stack. On '?', pop condition, true_val, false_val; push result. On ':', push. On value, push.

```python
def parseTernary(expression):
    st = []
    i = len(expression) - 1
    while i >= 0:
        c = expression[i]
        if c == '?':
            cond = st.pop()
            t = st.pop()
            f = st.pop()
            st.append(t if cond == 'T' else f)
        elif c != ':':
            st.append(c)
        i -= 1
    return st[0]
```

Time: O(n) | Space: O(n)

---

## 19. Tag Validator

Validate HTML/XML-like tags. Tags must be properly closed and nested. Stack for tag names. Parse open/close tags. Validate tag names (1-9 chars, uppercase). Match closing with top.

```python
def isValid(code):
    st = []
    i = 0
    while i < len(code):
        if i > 0 and not st:
            return False
        if code[i:i+9] == '<![CDATA[':
            j = code.find(']]>', i)
            if j == -1:
                return False
            i = j + 3
        elif code[i:i+2] == '</':
            j = code.find('>', i)
            tag = code[i+2:j]
            if not st or st[-1] != tag:
                return False
            st.pop()
            i = j + 1
        elif code[i] == '<':
            j = code.find('>', i)
            if j == -1:
                return False
            tag = code[i+1:j]
            if not (1 <= len(tag) <= 9 and tag.isupper()):
                return False
            st.append(tag)
            i = j + 1
        else:
            i += 1
    return len(st) == 0
```

Time: O(n) | Space: O(n)

---

## 20. Score of Parentheses

() = 1, AB = A+B, (A) = 2*A. Stack of scores. On '(', push 0. On ')', pop v, add max(2*v,1) to top.

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

## 21. Minimum Remove to Make Valid Parentheses

Remove minimum parentheses to make valid. Return any valid result. Stack of indices for '('. On ')', pop or mark remove. Remaining in stack: mark remove. Build excluding those.

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

---

## 22. Validate Stack Sequences

Given pushed and popped sequences, determine if valid. Simulate. Push from pushed. When top equals popped[j], pop and j++. Valid if stack empty at end.

```python
def validateStackSequences(pushed, popped):
    st = []
    j = 0
    for x in pushed:
        st.append(x)
        while st and st[-1] == popped[j]:
            st.pop()
            j += 1
    return len(st) == 0
```

Time: O(n) | Space: O(n)

---

## 23. Design Browser History

Back, forward, visit(url). Implement with stack (or doubly linked list). Two stacks: history (back) and forward. Visit: clear forward, push current to history. Back: push current to forward, pop from history. Forward: symmetric.

```python
class BrowserHistory:
    def __init__(self, homepage):
        self.history = [homepage]
        self.cur = 0

    def visit(self, url):
        self.history = self.history[:self.cur + 1] + [url]
        self.cur += 1

    def back(self, steps):
        self.cur = max(0, self.cur - steps)
        return self.history[self.cur]

    def forward(self, steps):
        self.cur = min(len(self.history) - 1, self.cur + steps)
        return self.history[self.cur]
```

Time: O(1) amortized | Space: O(n)

---

## 24. Design a Stack With Increment Operation

Stack with push, pop, and increment(k, val) which adds val to bottom k elements. Store (value, increment). On increment(k, val), add to min(k, size) bottom. On pop, add stored increment to result and propagate to next.

```python
class CustomStack:
    def __init__(self, maxSize):
        self.st = []
        self.inc = []
        self.maxSize = maxSize

    def push(self, x):
        if len(self.st) < self.maxSize:
            self.st.append(x)
            self.inc.append(0)

    def pop(self):
        if not self.st:
            return -1
        if len(self.inc) > 1:
            self.inc[-2] += self.inc[-1]
        return self.st.pop() + self.inc.pop()

    def increment(self, k, val):
        idx = min(k, len(self.st)) - 1
        if idx >= 0:
            self.inc[idx] += val
```

Time: O(1) | Space: O(n)

---

## 25. Maximum Frequency Stack

Push and pop. Pop returns most frequent element. Tie: most recent. Map freq. Map freq -> stack of elements. Track max_freq. Push: update freq, add to freq stack. Pop: pop from max_freq stack, update freq, decrement max_freq if empty.

```python
class FreqStack:
    def __init__(self):
        self.freq = {}
        self.group = {}
        self.max_freq = 0

    def push(self, val):
        self.freq[val] = self.freq.get(val, 0) + 1
        f = self.freq[val]
        self.max_freq = max(self.max_freq, f)
        if f not in self.group:
            self.group[f] = []
        self.group[f].append(val)

    def pop(self):
        x = self.group[self.max_freq].pop()
        self.freq[x] -= 1
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        return x
```

Time: O(1) | Space: O(n)

---

# Hard Problems

## 1. Largest Rectangle in Histogram

Find largest rectangle area in histogram. For each bar, find first smaller left and right. Monotonic increasing stack. Area = height * width.

```python
def largestRectangleArea(heights):
    st = []
    max_area = 0
    for i, h in enumerate(heights):
        while st and heights[st[-1]] > h:
            idx = st.pop()
            w = i - st[-1] - 1 if st else i
            max_area = max(max_area, heights[idx] * w)
        st.append(i)
    while st:
        idx = st.pop()
        w = len(heights) - st[-1] - 1 if st else len(heights)
        max_area = max(max_area, heights[idx] * w)
    return max_area
```

Time: O(n) | Space: O(n)

---

## 2. Maximal Rectangle

Binary matrix. Find largest rectangle of 1s. Treat each row as histogram base. Heights = consecutive 1s from top. Run largest rectangle per row.

```python
def maximalRectangle(matrix):
    if not matrix:
        return 0
    n, m = len(matrix), len(matrix[0])
    heights = [0] * (m + 1)
    max_area = 0
    for i in range(n):
        for j in range(m):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0
        st = []
        for j in range(m + 1):
            while st and heights[st[-1]] > heights[j]:
                idx = st.pop()
                w = j - st[-1] - 1 if st else j
                max_area = max(max_area, heights[idx] * w)
            st.append(j)
    return max_area
```

Time: O(n*m) | Space: O(m)

---

## 3. Trapping Rain Water

Compute how much water can be trapped between bars. Monotonic stack. When popping, water above popped bar = (min(current, stack[-1]) - popped_height) * width.

```python
def trap(height):
    st = []
    res = 0
    for i in range(len(height)):
        while st and height[st[-1]] < height[i]:
            mid = st.pop()
            if not st:
                break
            w = i - st[-1] - 1
            h = min(height[i], height[st[-1]]) - height[mid]
            res += w * h
        st.append(i)
    return res
```

Time: O(n) | Space: O(n)

---

## 4. Basic Calculator

Full calculator with +, -, *, /, parentheses, spaces. Recursive descent or two stacks. Handle parentheses with recursion/stack of states.

```python
def calculate(s):
    def calc(i):
        st, num, op = [], 0, '+'
        while i < len(s):
            c = s[i]
            if c.isdigit():
                num = num * 10 + int(c)
            elif c == '(':
                num, i = calc(i + 1)
            elif c in '+-*/':
                if op == '+': st.append(num)
                elif op == '-': st.append(-num)
                elif op == '*': st.append(st.pop() * num)
                else: st.append(int(st.pop() / num))
                op, num = c, 0
            elif c == ')':
                if op == '+': st.append(num)
                elif op == '-': st.append(-num)
                elif op == '*': st.append(st.pop() * num)
                else: st.append(int(st.pop() / num))
                return sum(st), i
            i += 1
        if op == '+': st.append(num)
        elif op == '-': st.append(-num)
        elif op == '*': st.append(st.pop() * num)
        else: st.append(int(st.pop() / num))
        return sum(st), i
    return calc(0)[0]
```

Time: O(n) | Space: O(n)

---

## 5. Longest Valid Parentheses

Find length of longest valid parentheses substring. Stack with -1. On '(', push index. On ')', pop; if empty push index else max_len = max(max_len, i - stack[-1]).

```python
def longestValidParentheses(s):
    st = [-1]
    max_len = 0
    for i, c in enumerate(s):
        if c == '(':
            st.append(i)
        else:
            st.pop()
            if not st:
                st.append(i)
            else:
                max_len = max(max_len, i - st[-1])
    return max_len
```

Time: O(n) | Space: O(n)

---

## 6. Remove Invalid Parentheses

Remove minimum number of parentheses to make valid. Return all possible results. BFS. Start with string. Generate all by removing one paren. Check valid. First valid level = answer. Deduplicate.

```python
def removeInvalidParentheses(s):
    def valid(x):
        bal = 0
        for c in x:
            if c == '(': bal += 1
            elif c == ')': bal -= 1
            if bal < 0: return False
        return bal == 0

    level = {s}
    while level:
        valid_res = [x for x in level if valid(x)]
        if valid_res:
            return valid_res
        level = {x[:i] + x[i+1:] for x in level for i in range(len(x)) if x[i] in '()'}
    return ['']
```

Time: O(2^n) | Space: O(n)

---

## 7. Number of Atoms

Parse formula like "K4(ON(SO3)2)2" and return count of each atom. Stack of counts. Parse atoms, numbers, '(', ')'. On ')', multiply by following number, merge into parent.

```python
def countOfAtoms(formula):
    from collections import Counter
    st = [Counter()]
    i, n = 0, len(formula)
    while i < n:
        if formula[i] == '(':
            st.append(Counter())
            i += 1
        elif formula[i] == ')':
            i += 1
            mult = 0
            while i < n and formula[i].isdigit():
                mult = mult * 10 + int(formula[i])
                i += 1
            mult = mult or 1
            top = st.pop()
            for k, v in top.items():
                st[-1][k] += v * mult
        else:
            atom = formula[i]
            i += 1
            while i < n and formula[i].islower():
                atom += formula[i]
                i += 1
            mult = 0
            while i < n and formula[i].isdigit():
                mult = mult * 10 + int(formula[i])
                i += 1
            st[-1][atom] += mult or 1
    return ''.join(f'{k}{v}' if v > 1 else k for k, v in sorted(st[-1].items()))
```

Time: O(n) | Space: O(n)

---

## 8. Decode String (Nested)

"3[a2[c]]" to "accaccacc". Stack. On ']', pop to '[', get k, push decoded.

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

## 9. Sum of Subarray Minimums

Sum of min over all subarrays. Mod 10^9+7. For each element as min, count subarrays. Left/right boundaries via monotonic stack. Contribution = arr[i] * (i-left) * (right-i).

```python
def sumSubarrayMins(arr):
    MOD = 10**9 + 7
    n = len(arr)
    left = [-1] * n
    right = [n] * n
    st = []
    for i in range(n):
        while st and arr[st[-1]] >= arr[i]:
            st.pop()
        if st:
            left[i] = st[-1]
        st.append(i)
    st = []
    for i in range(n - 1, -1, -1):
        while st and arr[st[-1]] > arr[i]:
            st.pop()
        if st:
            right[i] = st[-1]
        st.append(i)
    return sum(arr[i] * (i - left[i]) * (right[i] - i) for i in range(n)) % MOD
```

Time: O(n) | Space: O(n)

---

## 10. Sum of Subarray Ranges

Sum of (max - min) over all subarrays. Sum of subarray maximums minus sum of subarray minimums. Each uses monotonic stack for boundaries.

```python
def subArrayRanges(nums):
    def sum_sub(mul):
        st = []
        res = 0
        for i in range(len(nums) + 1):
            cur = nums[i] if i < len(nums) else float('inf') * mul
            while st and (nums[st[-1]] if mul == 1 else -nums[st[-1]]) < (cur if mul == 1 else -cur):
                idx = st.pop()
                left = st[-1] if st else -1
                res += nums[idx] * mul * (i - idx) * (idx - left)
            st.append(i)
        return res
    return sum_sub(1) - sum_sub(-1)
```

Time: O(n) | Space: O(n)

---

## 11. Maximum Width Ramp

Find max j-i such that A[i] <= A[j]. Decreasing stack of indices (by value). For each j from right, pop until stack top <= A[j], update width.

```python
def maxWidthRamp(nums):
    st = []
    for i in range(len(nums)):
        if not st or nums[st[-1]] > nums[i]:
            st.append(i)
    res = 0
    for j in range(len(nums) - 1, -1, -1):
        while st and nums[st[-1]] <= nums[j]:
            res = max(res, j - st.pop())
    return res
```

Time: O(n) | Space: O(n)

---

## 12. Constrained Subsequence Sum

Choose subsequence (no two adjacent) with max sum. At most k apart in original array. DP with deque for sliding window max. dp[i] = arr[i] + max(0, max(dp[i-k] to dp[i-1])).

```python
def constrainedSubsetSum(nums, k):
    from collections import deque
    dq = deque([0])
    dp = [0] * len(nums)
    for i in range(len(nums)):
        dp[i] = nums[i] + max(0, dp[dq[0]])
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
    return max(dp)
```

Time: O(n) | Space: O(n)

---

## 13. Sliding Window Maximum

For each window of size k, return maximum. Monotonic deque. Front = max. Pop back while back < current. Pop front if out of window.

```python
def maxSlidingWindow(nums, k):
    from collections import deque
    dq = deque()
    res = []
    for i in range(len(nums)):
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```

Time: O(n) | Space: O(k)

---

## 14. Minimum Cost Tree From Leaf Values

Given leaf values, build binary tree. Cost of node = product of max leaf in left and right subtree. Minimize sum of costs. Greedy with stack. Pop while top <= current (current is right max for popped). Cost += popped * min(current, top). Push current.

```python
def mctFromLeafValues(arr):
    st = [float('inf')]
    res = 0
    for x in arr:
        while st[-1] <= x:
            mid = st.pop()
            res += mid * min(st[-1], x)
        st.append(x)
    while len(st) > 2:
        res += st.pop() * st[-1]
    return res
```

Time: O(n) | Space: O(n)

---

## 15. Count of Smaller Numbers After Self

For each element, count elements to the right that are smaller. Merge sort with inversion count, or BST, or monotonic stack with binary search. Alternative: process right to left, maintain sorted list, binary search for count.

```python
def countSmaller(nums):
    import bisect
    sorted_nums = []
    res = []
    for x in reversed(nums):
        idx = bisect.bisect_left(sorted_nums, x)
        res.append(idx)
        bisect.insort(sorted_nums, x)
    return res[::-1]
```

Time: O(n log n) | Space: O(n)

---

## 16. Max Stack

Stack with push, pop, top, peekMax, popMax. popMax removes and returns the maximum element. Doubly linked list + treemap (or max heap). Or two stacks: main stack and max stack; popMax requires temporary stack to find and remove.

```python
class MaxStack:
    def __init__(self):
        self.st = []
        self.max_st = []

    def push(self, x):
        self.st.append(x)
        self.max_st.append(max(x, self.max_st[-1]) if self.max_st else x)

    def pop(self):
        self.max_st.pop()
        return self.st.pop()

    def top(self):
        return self.st[-1]

    def peekMax(self):
        return self.max_st[-1]

    def popMax(self):
        m = self.max_st[-1]
        buf = []
        while self.st[-1] != m:
            buf.append(self.pop())
        self.pop()
        while buf:
            self.push(buf.pop())
        return m
```

Time: O(n) for popMax, O(1) others | Space: O(n)

---

## 17. Expression Add Operators

Given digits and target, insert +, -, * to get target. Return all expressions. Backtracking. Track current value, previous operand for multiplication. On '*', subtract prev, add prev * current.

```python
def addOperators(num, target):
    res = []
    def backtrack(i, path, val, prev):
        if i == len(num):
            if val == target:
                res.append(path)
            return
        for j in range(i + 1, len(num) + 1):
            s = num[i:j]
            if s[0] == '0' and len(s) > 1:
                break
            n = int(s)
            if i == 0:
                backtrack(j, s, n, n)
            else:
                backtrack(j, path + '+' + s, val + n, n)
                backtrack(j, path + '-' + s, val - n, -n)
                backtrack(j, path + '*' + s, val - prev + prev * n, prev * n)
    backtrack(0, '', 0, 0)
    return res
```

Time: O(4^n) | Space: O(n)

---

## 18. Basic Calculator IV

Variables and numbers. Expand and simplify. Return list of terms. Parse to AST. Recursive descent. Handle +, -, *, parentheses. Simplify by merging like terms. Sort and format output.

```python
def basicCalculatorIV(expression, evalvars, evalints):
    from collections import Counter
    env = dict(zip(evalvars, evalints))
    def parse(s):
        s = s.replace(' ', '')
        def mul(a, b):
            return Counter({tuple(sorted((*k1, *k2))): v1 * v2 for (k1, v1) in a.items() for (k2, v2) in b.items()})
        def add(a, b):
            c = Counter(a)
            for k, v in b.items():
                c[k] += v
            return {k: v for k, v in c.items() if v}
        def eval_expr(s):
            stack = [{}]
            i = 0
            while i < len(s):
                if s[i] == ' ':
                    i += 1
                elif s[i] == '(':
                    stack.append({})
                    i += 1
                elif s[i] == ')':
                    top = stack.pop()
                    stack[-1] = add(stack[-1], top)
                    i += 1
                elif s[i] in '+':
                    i += 1
                elif s[i] == '-':
                    stack.append({(): -1})
                    i += 1
                elif s[i] == '*':
                    i += 1
                else:
                    j = i
                    while j < len(s) and s[j] not in ' ()+-*':
                        j += 1
                    tok = s[i:j]
                    if tok.isdigit():
                        stack.append({(): int(tok)})
                    elif tok in env:
                        stack.append({(): env[tok]})
                    else:
                        stack.append({((tok,),): 1})
                    i = j
            return stack[0]
        return eval_expr(s)
    poly = parse(expression)
    def format_term(k, v):
        if not k:
            return str(v)
        return '*'.join(k) + (f'*{v}' if v != 1 else '')
    terms = [format_term(k, v) for k, v in sorted(poly.items(), key=lambda x: (-len(x[0]), x[0])) if v]
    return terms
```

Time: O(n) | Space: O(n)
