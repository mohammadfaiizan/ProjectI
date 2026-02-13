# Sliding Window Technique

Maintain a window [left, right] that slides across the array. Expand right to include new elements, shrink left when constraint is violated. Used for subarray/substring problems with size or constraint limits.

## Max Sum Subarray of Size k

Fixed size window. Compute sum of first k, then slide: subtract left, add right. Time O(n), Space O(1).

```python
def max_sum_subarray_k(arr, k):
    if len(arr) < k:
        return 0
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

## Min Sum of Size k

Same as max but minimize. Time O(n), Space O(1).

```python
def min_sum_subarray_k(arr, k):
    if len(arr) < k:
        return 0
    window_sum = sum(arr[:k])
    min_sum = window_sum
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        min_sum = min(min_sum, window_sum)
    return min_sum
```

## Sliding Window Maximum

Use monotonic deque storing indices. Front is max of current window. Remove indices outside window, remove back elements smaller than current. Time O(n), Space O(k).

```python
def sliding_window_maximum(arr, k):
    from collections import deque
    dq = deque()
    result = []
    for i in range(len(arr)):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and arr[dq[-1]] <= arr[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result
```

## Sliding Window Minimum

Same as maximum but maintain increasing deque (remove back elements larger than current). Time O(n), Space O(k).

```python
def sliding_window_minimum(arr, k):
    from collections import deque
    dq = deque()
    result = []
    for i in range(len(arr)):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and arr[dq[-1]] >= arr[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result
```

## Count Distinct in Window k

Use hashmap for frequency. When window slides, decrement left element, increment right. Count distinct when freq becomes 1 or 0. Time O(n), Space O(k).

```python
def count_distinct_window(arr, k):
    from collections import defaultdict
    freq = defaultdict(int)
    distinct = 0
    result = []
    for i in range(len(arr)):
        if freq[arr[i]] == 0:
            distinct += 1
        freq[arr[i]] += 1
        if i >= k:
            freq[arr[i - k]] -= 1
            if freq[arr[i - k]] == 0:
                distinct -= 1
        if i >= k - 1:
            result.append(distinct)
    return result
```

## First Negative in Window k

Use deque to store indices of negative numbers in window. Front is first negative. Time O(n), Space O(k).

```python
def first_negative_window(arr, k):
    from collections import deque
    dq = deque()
    result = []
    for i in range(len(arr)):
        if arr[i] < 0:
            dq.append(i)
        while dq and dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(arr[dq[0]] if dq else 0)
    return result
```

## Smallest Subarray Sum At Least k

Variable size window. Expand right, when sum >= k try shrinking left. Time O(n), Space O(1).

```python
def smallest_subarray_sum_at_least_k(arr, k):
    left = total = 0
    min_len = float('inf')
    for right in range(len(arr)):
        total += arr[right]
        while total >= k and left <= right:
            min_len = min(min_len, right - left + 1)
            total -= arr[left]
            left += 1
    return min_len if min_len != float('inf') else 0
```

## Longest with Sum At Most k

Variable window. Expand right, shrink left when sum > k. Track max length. Time O(n), Space O(1).

```python
def longest_subarray_sum_at_most_k(arr, k):
    left = total = 0
    max_len = 0
    for right in range(len(arr)):
        total += arr[right]
        while total > k and left <= right:
            total -= arr[left]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Longest with At Most k Distinct

Variable window with frequency map. Expand right, shrink left when distinct > k. Time O(n), Space O(k).

```python
def longest_at_most_k_distinct(arr, k):
    from collections import defaultdict
    freq = defaultdict(int)
    left = distinct = max_len = 0
    for right in range(len(arr)):
        if freq[arr[right]] == 0:
            distinct += 1
        freq[arr[right]] += 1
        while distinct > k:
            freq[arr[left]] -= 1
            if freq[arr[left]] == 0:
                distinct -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Longest with At Most k Zeros (Max Consecutive Ones III)

Binary array. Flip at most k zeros to ones. Same as longest subarray with at most k zeros. Time O(n), Space O(1).

```python
def longest_ones(arr, k):
    left = zeros = max_len = 0
    for right in range(len(arr)):
        if arr[right] == 0:
            zeros += 1
        while zeros > k:
            if arr[left] == 0:
                zeros -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Fruit Into Baskets

At most 2 distinct types. Same as longest subarray with at most 2 distinct. Time O(n), Space O(1).

```python
def total_fruit(fruits):
    return longest_at_most_k_distinct(fruits, 2)
```

## Min Window Substring

Find minimum window in s containing all chars of t. Expand right, when valid shrink left. Use frequency maps. Time O(n), Space O(1) for 26 chars.

```python
def min_window(s, t):
    from collections import defaultdict
    need = defaultdict(int)
    for c in t:
        need[c] += 1
    have = 0
    required = len(need)
    window = defaultdict(int)
    left = 0
    result = ""
    min_len = float('inf')
    for right in range(len(s)):
        c = s[right]
        window[c] += 1
        if window[c] == need[c]:
            have += 1
        while have == required:
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = s[left:right + 1]
            window[s[left]] -= 1
            if window[s[left]] < need[s[left]]:
                have -= 1
            left += 1
    return result
```

## Permutation in String

Check if s2 contains permutation of s1. Use fixed window of len(s1) and compare frequency. Time O(n), Space O(1).

```python
def check_inclusion(s1, s2):
    from collections import Counter
    if len(s1) > len(s2):
        return False
    need = Counter(s1)
    window = Counter(s2[:len(s1)])
    if window == need:
        return True
    for i in range(len(s1), len(s2)):
        window[s2[i]] += 1
        window[s2[i - len(s1)]] -= 1
        if window[s2[i - len(s1)]] == 0:
            del window[s2[i - len(s1)]]
        if window == need:
            return True
    return False
```

## Find All Anagrams

Find start indices of anagrams of p in s. Same as permutation, collect indices. Time O(n), Space O(1).

```python
def find_anagrams(s, p):
    from collections import Counter
    result = []
    if len(p) > len(s):
        return result
    need = Counter(p)
    window = Counter(s[:len(p)])
    if window == need:
        result.append(0)
    for i in range(len(p), len(s)):
        window[s[i]] += 1
        window[s[i - len(p)]] -= 1
        if window[s[i - len(p)]] == 0:
            del window[s[i - len(p)]]
        if window == need:
            result.append(i - len(p) + 1)
    return result
```

## Longest Repeating Character Replacement

Replace at most k chars to get longest same-char substring. Window is valid when (window_size - max_freq) <= k. Time O(n), Space O(1).

```python
def character_replacement(s, k):
    from collections import defaultdict
    freq = defaultdict(int)
    left = max_len = max_freq = 0
    for right in range(len(s)):
        freq[s[right]] += 1
        max_freq = max(max_freq, freq[s[right]])
        while (right - left + 1) - max_freq > k:
            freq[s[left]] -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Max Vowels in Substring k

Fixed window of size k. Count vowels. Time O(n), Space O(1).

```python
def max_vowels(s, k):
    vowels = set('aeiou')
    count = sum(1 for c in s[:k] if c in vowels)
    max_count = count
    for i in range(k, len(s)):
        if s[i] in vowels:
            count += 1
        if s[i - k] in vowels:
            count -= 1
        max_count = max(max_count, count)
    return max_count
```

## Grumpy Bookstore Owner

Array of customers and grumpy. If grumpy[i]=1, owner is grumpy. Can use X minutes of not grumpy. Maximize satisfied customers. Sliding window for best X-minute segment to convert. Time O(n), Space O(1).

```python
def max_satisfied(customers, grumpy, minutes):
    base = sum(c for c, g in zip(customers, grumpy) if g == 0)
    extra = sum(customers[i] for i in range(minutes) if grumpy[i] == 1)
    max_extra = extra
    for i in range(minutes, len(customers)):
        if grumpy[i] == 1:
            extra += customers[i]
        if grumpy[i - minutes] == 1:
            extra -= customers[i - minutes]
        max_extra = max(max_extra, extra)
    return base + max_extra
```

## Max Points From Cards

Take k cards from either end. Equivalent to minimize sum of n-k consecutive cards in middle. Sliding window of size n-k. Time O(n), Space O(1).

```python
def max_score(card_points, k):
    n = len(card_points)
    window_size = n - k
    window_sum = sum(card_points[:window_size])
    min_sum = window_sum
    for i in range(window_size, n):
        window_sum = window_sum - card_points[i - window_size] + card_points[i]
        min_sum = min(min_sum, window_sum)
    return sum(card_points) - min_sum
```

## Get Equal Substrings Within Budget

Change s to t with cost per position. Longest substring with cost <= maxCost. Variable window. Time O(n), Space O(1).

```python
def equal_substring(s, t, max_cost):
    left = cost = max_len = 0
    for right in range(len(s)):
        cost += abs(ord(s[right]) - ord(t[right]))
        while cost > max_cost:
            cost -= abs(ord(s[left]) - ord(t[left]))
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len
```

## Frequency of Most Frequent Element

At most k increments. Maximize frequency of most frequent element. Sort, sliding window: valid when (window_size * max_val - window_sum) <= k. Time O(n log n), Space O(1).

```python
def max_frequency(arr, k):
    arr = sorted(arr)
    left = total = max_freq = 0
    for right in range(len(arr)):
        total += arr[right]
        while (right - left + 1) * arr[right] - total > k:
            total -= arr[left]
            left += 1
        max_freq = max(max_freq, right - left + 1)
    return max_freq
```

## Longest Subarray of 1s After Deleting One

At most one 0. Same as longest subarray with at most 1 zero. Time O(n), Space O(1).

```python
def longest_subarray_ones(arr):
    left = zeros = max_len = 0
    for right in range(len(arr)):
        if arr[right] == 0:
            zeros += 1
        while zeros > 1:
            if arr[left] == 0:
                zeros -= 1
            left += 1
        max_len = max(max_len, right - left)
    return max_len
```

## Count Nice Subarrays

Nice = odd count is exactly k. Count subarrays with at most k odd minus at most k-1 odd. Time O(n), Space O(1).

```python
def count_nice_subarrays(arr, k):
    def at_most(limit):
        left = odds = count = 0
        for right in range(len(arr)):
            if arr[right] % 2 == 1:
                odds += 1
            while odds > limit:
                if arr[left] % 2 == 1:
                    odds -= 1
                left += 1
            count += right - left + 1
        return count
    return at_most(k) - at_most(k - 1)
```

## Binary Subarrays With Sum

Count subarrays with sum goal. Prefix + hashmap or at_most(goal) - at_most(goal-1). Time O(n), Space O(n).

```python
def num_subarrays_with_sum(arr, goal):
    def at_most(limit):
        if limit < 0:
            return 0
        left = total = count = 0
        for right in range(len(arr)):
            total += arr[right]
            while total > limit:
                total -= arr[left]
                left += 1
            count += right - left + 1
        return count
    return at_most(goal) - at_most(goal - 1)
```

## Subarrays with K Different Integers

Count subarrays with exactly k distinct. At most k minus at most k-1. Time O(n), Space O(k).

```python
def subarrays_with_k_different(arr, k):
    def at_most(limit):
        from collections import defaultdict
        freq = defaultdict(int)
        left = distinct = count = 0
        for right in range(len(arr)):
            if freq[arr[right]] == 0:
                distinct += 1
            freq[arr[right]] += 1
            while distinct > limit:
                freq[arr[left]] -= 1
                if freq[arr[left]] == 0:
                    distinct -= 1
                left += 1
            count += right - left + 1
        return count
    return at_most(k) - at_most(k - 1)
```

## Shortest Subarray Sum At Least k (Deque)

Array can have negatives. Monotonic deque of prefix sums. For each j, find largest i with prefix[j] - prefix[i] >= k. Time O(n), Space O(n).

```python
def shortest_subarray_deque(arr, k):
    n = len(arr)
    prefix = [0]
    for x in arr:
        prefix.append(prefix[-1] + x)
    from collections import deque
    dq = deque()
    result = n + 1
    for j in range(n + 1):
        while dq and prefix[j] - prefix[dq[0]] >= k:
            result = min(result, j - dq.popleft())
        while dq and prefix[dq[-1]] >= prefix[j]:
            dq.pop()
        dq.append(j)
    return result if result <= n else -1
```
