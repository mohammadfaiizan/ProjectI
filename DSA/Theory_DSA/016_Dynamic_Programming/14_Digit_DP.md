# Digit DP

## Concept

Digit DP counts numbers in range [L, R] satisfying digit constraints. Process digits from left to right. Key state components:
- **position**: current digit index
- **tight**: whether prefix equals the upper bound prefix (limits digit choices)
- **leading_zeros**: whether we have leading zeros (for constraints on zero)
- **additional flags**: problem-specific (e.g., digit seen, sum, etc.)

When tight is True, the next digit cannot exceed the corresponding digit of the upper bound. When we choose a smaller digit, tight becomes False for all subsequent positions.

## Count Numbers with No Repeated Digits

```python
def count_numbers_no_repeat(n):
    s = str(n)
    
    def dp(pos, tight, mask):
        if pos == len(s):
            return 1
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in range(limit + 1):
            if mask & (1 << d):
                continue
            new_tight = tight and (d == limit)
            new_mask = mask | (1 << d) if (mask or d) else mask
            total += dp(pos + 1, new_tight, new_mask)
        return total
    
    return dp(0, True, 0)
```

## Numbers At Most N Given Digit Set

```python
def at_most_n_given_digit_set(digits, n):
    s = str(n)
    
    def dp(pos, tight):
        if pos == len(s):
            return 1
        limit = int(s[pos]) if tight else 9
        total = 0
        for d in digits:
            if d > limit:
                break
            new_tight = tight and (d == limit)
            total += dp(pos + 1, new_tight)
        return total
    
    result = 0
    for length in range(1, len(s)):
        result += len(digits) ** length
    result += dp(0, True)
    return result
```

## Count Special Integers (All Distinct)

```python
def count_special_integers(n):
    s = str(n)
    
    def dp(pos, tight, mask, started):
        if pos == len(s):
            return 1 if started else 0
        limit = int(s[pos]) if tight else 9
        total = 0
        start = 0 if started else 1
        for d in range(start, limit + 1):
            if mask & (1 << d):
                continue
            new_tight = tight and (d == limit)
            new_mask = mask | (1 << d)
            new_started = started or (d > 0)
            total += dp(pos + 1, new_tight, new_mask, new_started)
        if not started:
            total += dp(pos + 1, False, mask, False)
        return total
    
    return dp(0, True, 0, False)
```

## Non-Negative Integers Without Consecutive Ones

```python
def find_integers(n):
    s = bin(n)[2:]
    
    def dp(pos, tight, prev_one):
        if pos == len(s):
            return 1
        limit = int(s[pos]) if tight else 1
        total = 0
        for d in range(limit + 1):
            if prev_one and d == 1:
                continue
            new_tight = tight and (d == limit)
            total += dp(pos + 1, new_tight, d == 1)
        return total
    
    return dp(0, True, False)
```

## Count Stepping Numbers in Range

```python
def count_stepping_numbers(low, high):
    def count_up_to(n):
        if n < 0:
            return 0
        s = str(n)
        memo = {}
        
        def dp(pos, tight, prev):
            if pos == len(s):
                return 1
            key = (pos, tight, prev)
            if key in memo:
                return memo[key]
            limit = int(s[pos]) if tight else 9
            total = 0
            start = 1 if prev is None else 0
            for d in range(start, limit + 1):
                if prev is not None and abs(d - prev) != 1:
                    continue
                new_tight = tight and (d == limit)
                total += dp(pos + 1, new_tight, d)
            memo[key] = total
            return total
        
        result = dp(0, True, None)
        for length in range(1, len(s)):
            for first in range(1, 10):
                result += count_shorter(length, first)
        return result
    
    def count_shorter(length, prev):
        if length == 1:
            return 1
        total = 0
        for d in [prev - 1, prev + 1]:
            if 0 <= d <= 9:
                total += count_shorter(length - 1, d)
        return total
    
    return count_up_to(high) - count_up_to(low - 1)
```

## Digit DP Template (General)

```python
def digit_dp_template(upper_bound_str, constraint_check):
    memo = {}
    
    def dp(pos, tight, *state):
        if pos == len(upper_bound_str):
            return 1 if constraint_check(*state) else 0
        key = (pos, tight) + state
        if key in memo:
            return memo[key]
        limit = int(upper_bound_str[pos]) if tight else 9
        total = 0
        for d in range(limit + 1):
            new_tight = tight and (d == limit)
            new_state = update_state(pos, d, *state)
            if is_valid(new_state):
                total += dp(pos + 1, new_tight, *new_state)
        memo[key] = total
        return total
    
    return dp(0, True, *initial_state())
```
