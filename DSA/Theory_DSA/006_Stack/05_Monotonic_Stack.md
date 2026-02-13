# Monotonic Stack

## Theory

A monotonic stack maintains elements in strictly increasing or strictly decreasing order. Used to find next greater/smaller element in O(n) time. For each index, we pop elements that violate the monotonic property and record the relationship (e.g., popped element's next greater is current element).

**Monotonic increasing stack**: Top is smallest. Pop while current < top when finding next smaller.
**Monotonic decreasing stack**: Top is largest. Pop while current > top when finding next greater.

## Next Greater Element to Right

For each element, find the first element to its right that is greater. Monotonic decreasing stack: store indices. Pop when current > arr[stack[-1]]; for popped index, NGE is current.

```python
def next_greater_right(arr):
    n = len(arr)
    result = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] < arr[i]:
            result[stack.pop()] = arr[i]
        stack.append(i)
    return result
```

## Next Greater Element to Left

For each element, find the first element to its left that is greater. Process left to right, stack stores indices. Pop when current > arr[stack[-1]]; for popped, NGE-left is current. Then push current.

```python
def next_greater_left(arr):
    n = len(arr)
    result = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] < arr[i]:
            stack.pop()
        if stack:
            result[i] = arr[stack[-1]]
        stack.append(i)
    return result
```

## Next Smaller Element to Right

Monotonic increasing stack: pop when current < arr[stack[-1]]. For popped index, NSE-right is current.

```python
def next_smaller_right(arr):
    n = len(arr)
    result = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] > arr[i]:
            result[stack.pop()] = arr[i]
        stack.append(i)
    return result
```

## Next Smaller Element to Left

For each element, first smaller to the left. Stack stores indices. Pop when current < arr[stack[-1]]. If stack non-empty, NSE-left is arr[stack[-1]].

```python
def next_smaller_left(arr):
    n = len(arr)
    result = [-1] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        if stack:
            result[i] = arr[stack[-1]]
        stack.append(i)
    return result
```

## Next Greater in Circular Array (NGE II)

Array is circular. For each element, find next greater (may wrap). Double the array (or use modulo) and run NGE. Only consider first n elements in result.

```python
def next_greater_circular(arr):
    n = len(arr)
    result = [-1] * n
    stack = []
    for i in range(2 * n):
        idx = i % n
        while stack and arr[stack[-1]] < arr[idx]:
            result[stack.pop()] = arr[idx]
        if i < n:
            stack.append(idx)
    return result
```

## Stock Span Problem

Span of day i = 1 + number of consecutive days to the left where price was <= price[i]. Monotonic decreasing stack of indices. Pop while price[stack[-1]] <= price[i]. Span = i - stack[-1] if stack else i + 1.

```python
def stock_span(prices):
    n = len(prices)
    span = [0] * n
    stack = []
    for i in range(n):
        while stack and prices[stack[-1]] <= prices[i]:
            stack.pop()
        span[i] = i - stack[-1] if stack else i + 1
        stack.append(i)
    return span
```

## Daily Temperatures

For each day, find number of days until a warmer day. NGE variant: store (index, value). When we find next greater for stack top, result[stack_top] = i - stack_top.

```python
def daily_temperatures(temperatures):
    n = len(temperatures)
    result = [0] * n
    stack = []
    for i in range(n):
        while stack and temperatures[stack[-1]] < temperatures[i]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    return result
```

## Largest Rectangle in Histogram

For each bar, find largest rectangle with that bar as height. Need left and right boundaries (first smaller to left and right). Area = height * (right_smaller_idx - left_smaller_idx - 1). Use monotonic increasing stack.

```python
def largest_rectangle_histogram(heights):
    n = len(heights)
    left = [-1] * n
    right = [n] * n
    stack = []
    for i in range(n):
        while stack and heights[stack[-1]] >= heights[i]:
            stack.pop()
        if stack:
            left[i] = stack[-1]
        stack.append(i)
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and heights[stack[-1]] >= heights[i]:
            stack.pop()
        if stack:
            right[i] = stack[-1]
        stack.append(i)
    max_area = 0
    for i in range(n):
        max_area = max(max_area, heights[i] * (right[i] - left[i] - 1))
    return max_area
```

Single pass variant:

```python
def largest_rectangle_histogram_single_pass(heights):
    stack = []
    max_area = 0
    heights.append(0)
    for i in range(len(heights)):
        while stack and heights[stack[-1]] > heights[i]:
            h = heights[stack.pop()]
            w = i - stack[-1] - 1 if stack else i
            max_area = max(max_area, h * w)
        stack.append(i)
    return max_area
```

## Maximal Rectangle in Binary Matrix (Histogram Per Row)

Treat each row as base of histogram. Heights = consecutive 1s from top. Run largest rectangle for each row.

```python
def maximal_rectangle(matrix):
    if not matrix:
        return 0
    m, n = len(matrix), len(matrix[0])
    heights = [0] * (n + 1)
    max_area = 0
    for row in matrix:
        for j in range(n):
            heights[j] = heights[j] + 1 if row[j] == '1' else 0
        stack = []
        for i in range(n + 1):
            while stack and heights[stack[-1]] > heights[i]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1 if stack else i
                max_area = max(max_area, h * w)
            stack.append(i)
    return max_area
```

## Trapping Rain Water Using Stack

For each bar, water trapped on top = min(max_left, max_right) - height. Stack approach: maintain decreasing stack. When we pop (bar at idx), water trapped above it is (min(current, stack[-1]) - popped_height) * width. Width = i - stack[-1] - 1.

```python
def trap_rain_water(height):
    stack = []
    water = 0
    for i in range(len(height)):
        while stack and height[stack[-1]] < height[i]:
            top = stack.pop()
            if not stack:
                break
            dist = i - stack[-1] - 1
            h = min(height[i], height[stack[-1]]) - height[top]
            water += dist * h
        stack.append(i)
    return water
```

## Sum of Subarray Minimums

For each element, count subarrays where it is the minimum. Left: first smaller to left (exclusive). Right: first smaller to right (exclusive). Count = (i - left) * (right - i). Sum += arr[i] * count.

```python
def sum_subarray_minimums(arr):
    n = len(arr)
    left = [-1] * n
    right = [n] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        if stack:
            left[i] = stack[-1]
        stack.append(i)
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        if stack:
            right[i] = stack[-1]
        stack.append(i)
    mod = 10**9 + 7
    total = 0
    for i in range(n):
        total = (total + arr[i] * (i - left[i]) * (right[i] - i)) % mod
    return total
```

## Sum of Subarray Maximums

Same idea as minimums but for maximum. Use next greater to left and right.

```python
def sum_subarray_maximums(arr):
    n = len(arr)
    left = [-1] * n
    right = [n] * n
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] < arr[i]:
            stack.pop()
        if stack:
            left[i] = stack[-1]
        stack.append(i)
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] <= arr[i]:
            stack.pop()
        if stack:
            right[i] = stack[-1]
        stack.append(i)
    mod = 10**9 + 7
    total = 0
    for i in range(n):
        total = (total + arr[i] * (i - left[i]) * (right[i] - i)) % mod
    return total
```

## Sum of Subarray Ranges

Sum of (max - min) over all subarrays. Equals sum of subarray maximums minus sum of subarray minimums.

```python
def sum_subarray_ranges(arr):
    return sum_subarray_maximums(arr) - sum_subarray_minimums(arr)
```

## Maximum Width Ramp

Ramp: i < j and A[i] <= A[j]. Width = j - i. Find max width. Maintain decreasing stack of indices (by value). For each j from right, pop until stack top <= A[j], update width.

```python
def max_width_ramp(arr):
    stack = []
    for i in range(len(arr)):
        if not stack or arr[stack[-1]] > arr[i]:
            stack.append(i)
    max_width = 0
    for j in range(len(arr) - 1, -1, -1):
        while stack and arr[stack[-1]] <= arr[j]:
            max_width = max(max_width, j - stack.pop())
    return max_width
```

## Remove K Digits to Make Smallest

Given number as string, remove k digits to get smallest number. Monotonic increasing stack: pop while top > current and we have removals left.

```python
def remove_k_digits(num, k):
    stack = []
    for d in num:
        while k and stack and stack[-1] > d:
            stack.pop()
            k -= 1
        stack.append(d)
    while k:
        stack.pop()
        k -= 1
    result = ''.join(stack).lstrip('0')
    return result if result else '0'
```

## Remove Duplicate Letters (Smallest Subsequence)

Remove duplicates and get lexicographically smallest. Stack: push char. Pop while top > current, top appears later, and we haven't used last occurrence of top.

```python
def remove_duplicate_letters(s):
    last = {c: i for i, c in enumerate(s)}
    seen = set()
    stack = []
    for i, c in enumerate(s):
        if c in seen:
            continue
        while stack and stack[-1] > c and last[stack[-1]] > i:
            seen.discard(stack.pop())
        stack.append(c)
        seen.add(c)
    return ''.join(stack)
```

## Most Competitive Subsequence

Length k, lexicographically smallest. Monotonic stack: pop while top > current and remaining elements + stack size > k.

```python
def most_competitive(nums, k):
    stack = []
    n = len(nums)
    for i, x in enumerate(nums):
        while stack and stack[-1] > x and len(stack) + (n - i) > k:
            stack.pop()
        if len(stack) < k:
            stack.append(x)
    return stack
```

## 132 Pattern

Find i < j < k with arr[i] < arr[k] < arr[j]. Traverse from right. Maintain stack (decreasing). Keep track of third (arr[k]). When we see arr[i] < third, return True.

```python
def find132pattern(nums):
    third = float('-inf')
    stack = []
    for x in reversed(nums):
        if x < third:
            return True
        while stack and stack[-1] < x:
            third = stack.pop()
        stack.append(x)
    return False
```

## Asteroid Collision

Positive = right, negative = left. Collide when left-moving (negative) meets right-moving (positive). Smaller explodes. Same size both explode. Stack: push positive. For negative, pop while stack and stack[-1] > 0 and stack[-1] < abs(neg). If stack[-1] == abs(neg), pop and skip. If stack empty or top < 0, push negative.

```python
def asteroid_collision(asteroids):
    stack = []
    for a in asteroids:
        if a > 0:
            stack.append(a)
        else:
            while stack and stack[-1] > 0 and stack[-1] < -a:
                stack.pop()
            if stack and stack[-1] == -a:
                stack.pop()
            elif not stack or stack[-1] < 0:
                stack.append(a)
    return stack
```

## Online Stock Span

Stream of prices. For each new price, return span (consecutive days with price <= today). Maintain monotonic decreasing stack of (price, span). Pop while top <= current, add spans. Push (price, total_span).

```python
class StockSpanner:
    def __init__(self):
        self.stack = []

    def next(self, price):
        span = 1
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        self.stack.append((price, span))
        return span
```

## Car Fleet

Cars at position and speed. They form fleets when faster car catches slower. Sort by position descending. Monotonic stack: time to target = (target - pos) / speed. Pop while stack and time[stack[-1]] <= time[current] (current catches up). Push current.

```python
def car_fleet(target, position, speed):
    cars = sorted(zip(position, speed), reverse=True)
    stack = []
    for pos, spd in cars:
        t = (target - pos) / spd
        while stack and t >= stack[-1]:
            stack.pop()
        stack.append(t)
    return len(stack)
```

## Number of Visible People in Queue

Person i can see person j (i < j) if everyone between is shorter than both. For each i, count visible to the right. Monotonic decreasing stack: when we pop, those are visible from current. Count = number of pops + 1 if stack non-empty.

```python
def can_see_persons_count(heights):
    n = len(heights)
    result = [0] * n
    stack = []
    for i in range(n - 1, -1, -1):
        count = 0
        while stack and heights[stack[-1]] < heights[i]:
            stack.pop()
            count += 1
        result[i] = count + (1 if stack else 0)
        stack.append(i)
    return result
```

## Sum of Total Strength of Wizards

For each wizard as minimum in a subarray, contribution = strength[i] * (sum of all subarrays where i is min). Requires left/right boundaries (next smaller), then prefix sums of prefix sums for range sum queries.

```python
def total_strength(strength):
    n = len(strength)
    mod = 10**9 + 7
    left = [-1] * n
    right = [n] * n
    stack = []
    for i in range(n):
        while stack and strength[stack[-1]] >= strength[i]:
            stack.pop()
        if stack:
            left[i] = stack[-1]
        stack.append(i)
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and strength[stack[-1]] > strength[i]:
            stack.pop()
        if stack:
            right[i] = stack[-1]
        stack.append(i)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = (prefix[i] + strength[i]) % mod
    prefix_prefix = [0] * (n + 2)
    for i in range(n + 1):
        prefix_prefix[i + 1] = (prefix_prefix[i] + prefix[i]) % mod

    total = 0
    for i in range(n):
        L, R = left[i] + 1, right[i] - 1
        left_sum = (prefix_prefix[i + 1] - prefix_prefix[L] - prefix[L] * (i - L + 1)) % mod
        right_sum = (prefix[R + 1] * (R - i + 1) - (prefix_prefix[R + 2] - prefix_prefix[i + 2])) % mod
        total = (total + strength[i] * ((left_sum * (R - i + 1) + right_sum * (i - L + 1)) % mod)) % mod
    return total % mod
```
