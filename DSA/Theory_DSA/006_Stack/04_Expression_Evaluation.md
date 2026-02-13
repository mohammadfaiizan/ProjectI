# Expression Evaluation

## Infix to Postfix (Shunting-Yard with Precedence and Associativity)

Shunting-yard algorithm: scan left to right. Operands go to output. Operators go to stack: pop operators with higher precedence (or equal if left-associative) to output before pushing. Left paren: push. Right paren: pop to output until left paren.

Precedence: ^ (right-assoc) > * / > + -
Associativity: ^ right-to-left; * / + - left-to-right.

```python
def precedence(op):
    if op in '+-':
        return 1
    if op in '*/':
        return 2
    if op == '^':
        return 3
    return 0

def is_right_assoc(op):
    return op == '^'

def infix_to_postfix(expr):
    output = []
    stack = []
    for c in expr:
        if c.isalnum():
            output.append(c)
        elif c == '(':
            stack.append(c)
        elif c == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()
        elif c in '+-*/^':
            while stack and stack[-1] != '(':
                top = stack[-1]
                if top in '+-*/^':
                    if is_right_assoc(c):
                        if precedence(c) < precedence(top):
                            output.append(stack.pop())
                        else:
                            break
                    else:
                        if precedence(c) <= precedence(top):
                            output.append(stack.pop())
                        else:
                            break
                else:
                    break
            stack.append(c)
    while stack:
        output.append(stack.pop())
    return ''.join(output)
```

## Infix to Prefix

Reverse the expression, swap '(' and ')', apply infix-to-postfix logic, then reverse the result. Alternatively: build postfix first, then reverse the output string (operands stay, operators reversed in order).

```python
def infix_to_prefix(expr):
    expr = expr[::-1]
    rev = []
    for c in expr:
        if c == '(':
            rev.append(')')
        elif c == ')':
            rev.append('(')
        else:
            rev.append(c)
    postfix = infix_to_postfix(''.join(rev))
    return postfix[::-1]
```

## Postfix Evaluation

Scan left to right. Operands: push. Operator: pop two operands, apply operator, push result. Final result is stack top.

```python
def eval_postfix(expr):
    stack = []
    for c in expr:
        if c.isdigit():
            stack.append(int(c))
        elif c in '+-*/':
            b = stack.pop()
            a = stack.pop()
            if c == '+':
                stack.append(a + b)
            elif c == '-':
                stack.append(a - b)
            elif c == '*':
                stack.append(a * b)
            elif c == '/':
                stack.append(int(a / b))
    return stack[-1]
```

For multi-digit numbers, tokenize by spaces:

```python
def eval_postfix_tokens(expr):
    stack = []
    for tok in expr.split():
        if tok.isdigit() or (tok[0] == '-' and tok[1:].isdigit()):
            stack.append(int(tok))
        else:
            b = stack.pop()
            a = stack.pop()
            if tok == '+':
                stack.append(a + b)
            elif tok == '-':
                stack.append(a - b)
            elif tok == '*':
                stack.append(a * b)
            elif tok == '/':
                stack.append(int(a / b))
    return stack[-1]
```

## Prefix Evaluation

Scan right to left. Operands: push. Operator: pop two (first popped is right operand, second is left), apply, push result.

```python
def eval_prefix(expr):
    stack = []
    for c in reversed(expr):
        if c.isdigit():
            stack.append(int(c))
        elif c in '+-*/':
            a = stack.pop()
            b = stack.pop()
            if c == '+':
                stack.append(a + b)
            elif c == '-':
                stack.append(a - b)
            elif c == '*':
                stack.append(a * b)
            elif c == '/':
                stack.append(int(a / b))
    return stack[-1]
```

## Infix Evaluation (With Parentheses and Precedence)

Two stacks: operands and operators. On operator: pop and apply while stack top has higher or equal precedence. On '(' push. On ')' pop and apply until '('.

```python
def apply_op(a, b, op):
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        return int(a / b)
    return 0

def precedence(op):
    if op in '+-':
        return 1
    if op in '*/':
        return 2
    return 0

def eval_infix(s):
    values = []
    ops = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == ' ':
            i += 1
            continue
        if s[i] == '(':
            ops.append(s[i])
            i += 1
        elif s[i].isdigit():
            num = 0
            while i < n and s[i].isdigit():
                num = num * 10 + int(s[i])
                i += 1
            values.append(num)
        elif s[i] == ')':
            while ops and ops[-1] != '(':
                b, a = values.pop(), values.pop()
                values.append(apply_op(a, b, ops.pop()))
            ops.pop()
            i += 1
        elif s[i] in '+-*/':
            while ops and ops[-1] != '(' and precedence(ops[-1]) >= precedence(s[i]):
                b, a = values.pop(), values.pop()
                values.append(apply_op(a, b, ops.pop()))
            ops.append(s[i])
            i += 1
        else:
            i += 1
    while ops:
        b, a = values.pop(), values.pop()
        values.append(apply_op(a, b, ops.pop()))
    return values[-1]
```

## Basic Calculator I (+ - Parentheses)

Same as infix evaluation with + and - only. Precedence is equal, so left-to-right. Handle unary minus by treating it as 0 - x.

```python
def calculate_basic1(s):
    stack = []
    num = 0
    sign = 1
    result = 0
    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)
        elif c == '+':
            result += sign * num
            num = 0
            sign = 1
        elif c == '-':
            result += sign * num
            num = 0
            sign = -1
        elif c == '(':
            result += sign * num
            num = 0
            stack.append((result, sign))
            result = 0
            sign = 1
        elif c == ')':
            result += sign * num
            num = 0
            prev_result, prev_sign = stack.pop()
            result = prev_result + prev_sign * result
            sign = 1
    return result + sign * num
```

## Basic Calculator II (+ - * /)

No parentheses. Two passes or one pass with stack: push numbers with sign; when * or /, pop last number, apply, push result. Finally sum all.

```python
def calculate_basic2(s):
    stack = []
    num = 0
    op = '+'
    for i, c in enumerate(s):
        if c.isdigit():
            num = num * 10 + int(c)
        if c in '+-*/' or i == len(s) - 1:
            if c != ' ' or i == len(s) - 1:
                if op == '+':
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)
                elif op == '*':
                    stack.append(stack.pop() * num)
                elif op == '/':
                    stack.append(int(stack.pop() / num))
                num = 0
                if c in '+-*/':
                    op = c
    return sum(stack)
```

## Basic Calculator III (+ - * / Parentheses)

Combine I and II: when we see '(', push current result and sign; when ')', pop and combine. For * and /, apply immediately on last number.

```python
def calculate_basic3(s):
    def calc(i):
        stack = []
        num = 0
        op = '+'
        while i < len(s):
            c = s[i]
            if c.isdigit():
                num = num * 10 + int(c)
                i += 1
            elif c == '(':
                num, i = calc(i + 1)
                i += 1
            elif c == ')':
                if op == '+':
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)
                elif op == '*':
                    stack.append(stack.pop() * num)
                elif op == '/':
                    stack.append(int(stack.pop() / num))
                return sum(stack), i
            elif c in '+-*/':
                if op == '+':
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)
                elif op == '*':
                    stack.append(stack.pop() * num)
                elif op == '/':
                    stack.append(int(stack.pop() / num))
                num = 0
                op = c
                i += 1
            else:
                i += 1
        if op == '+':
            stack.append(num)
        elif op == '-':
            stack.append(-num)
        elif op == '*':
            stack.append(stack.pop() * num)
        elif op == '/':
            stack.append(int(stack.pop() / num))
        return sum(stack), i
    return calc(0)[0]
```

## Evaluate Reverse Polish Notation

Same as postfix evaluation. Tokens are operators or numbers.

```python
def eval_rpn(tokens):
    stack = []
    for t in tokens:
        if t in '+-*/':
            b = stack.pop()
            a = stack.pop()
            if t == '+':
                stack.append(a + b)
            elif t == '-':
                stack.append(a - b)
            elif t == '*':
                stack.append(a * b)
            elif t == '/':
                stack.append(int(a / b))
        else:
            stack.append(int(t))
    return stack[-1]
```

## Decode String (Nested Encoding 3[a2[c]])

Stack: push chars and '['. When ']', pop until '[', then pop digits for k, repeat string k times, push back.

```python
def decode_string(s):
    stack = []
    for c in s:
        if c != ']':
            stack.append(c)
        else:
            sub = []
            while stack[-1] != '[':
                sub.append(stack.pop())
            stack.pop()
            k = []
            while stack and stack[-1].isdigit():
                k.append(stack.pop())
            k = int(''.join(reversed(k))) if k else 1
            sub = ''.join(reversed(sub)) * k
            for ch in sub:
                stack.append(ch)
    return ''.join(stack)
```

## Remove Outermost Parentheses

Track depth. When depth goes 1->0, we see ')'. When 0->1, we see '('. Skip those. Otherwise append.

```python
def remove_outer_parentheses(s):
    result = []
    depth = 0
    for c in s:
        if c == '(':
            if depth > 0:
                result.append(c)
            depth += 1
        else:
            depth -= 1
            if depth > 0:
                result.append(c)
    return ''.join(result)
```

## Min Add to Make Parentheses Valid

Count unmatched open and unmatched close. Unmatched open: '(' without ')'. Unmatched close: ')' without '('. Use balance: +1 for '(', -1 for ')'. Negative means need to add '('. Final positive means need to add ')'.

```python
def min_add_to_make_valid(s):
    open_needed = 0
    close_needed = 0
    for c in s:
        if c == '(':
            close_needed += 1
        else:
            if close_needed > 0:
                close_needed -= 1
            else:
                open_needed += 1
    return open_needed + close_needed
```

## Min Remove for Valid Parentheses

Find indices to remove. Stack of indices for '('. On ')', if stack pop; else mark for removal. Remaining in stack: mark for removal. Build result excluding those indices.

```python
def min_remove_to_make_valid(s):
    stack = []
    remove = set()
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                stack.pop()
            else:
                remove.add(i)
    remove.update(stack)
    return ''.join(c for i, c in enumerate(s) if i not in remove)
```

## Score of Parentheses

() = 1. AB = A + B. (A) = 2 * A. Use stack: 0 for base. On '(', push 0. On ')', pop, score = max(2 * popped, 1), add to new top.

```python
def score_of_parentheses(s):
    stack = [0]
    for c in s:
        if c == '(':
            stack.append(0)
        else:
            v = stack.pop()
            stack[-1] += max(2 * v, 1)
    return stack[-1]
```

## Max Nesting Depth

Track current depth. Max depth = max over all '(' encounters.

```python
def max_depth(s):
    depth = 0
    max_d = 0
    for c in s:
        if c == '(':
            depth += 1
            max_d = max(max_d, depth)
        elif c == ')':
            depth -= 1
    return max_d
```

## Valid Parentheses (Multiple Types)

Stack: push '(', '[', '{'. On ')', ']', '}', pop and check match. Empty stack on closing is invalid.

```python
def valid_parentheses(s):
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for c in s:
        if c in '([{':
            stack.append(c)
        else:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return len(stack) == 0
```

## Longest Valid Parentheses

DP or stack. Stack approach: push -1 initially. For '(' push index. For ')' pop; if stack empty push index else length = i - stack[-1].

```python
def longest_valid_parentheses(s):
    stack = [-1]
    max_len = 0
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])
    return max_len
```

## Generate All Valid Parentheses

Backtrack: add '(' if open < n, add ')' if close < open. When open == close == n, record.

```python
def generate_parenthesis(n):
    result = []

    def backtrack(s, open_count, close_count):
        if len(s) == 2 * n:
            result.append(s)
            return
        if open_count < n:
            backtrack(s + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(s + ')', open_count, close_count + 1)

    backtrack('', 0, 0)
    return result
```

## Check Redundant Braces

Redundant if we have (a) or (a+b) where the inner expr has no operator. Stack: push until ')'. When ')', pop until '('. If we see an operator between, not redundant. If no operator, redundant.

```python
def has_redundant_braces(s):
    stack = []
    for c in s:
        if c == ')':
            has_op = False
            while stack and stack[-1] != '(':
                if stack[-1] in '+-*/':
                    has_op = True
                stack.pop()
            stack.pop()
            if not has_op:
                return True
        else:
            stack.append(c)
    return False
```

## Count Reversals to Balance

Count open and close. Each reversal fixes one. Need open and close to be even. If total odd, return -1. Reversals = ceil(open/2) + ceil(close/2). Track balance: for '{' +1, for '}' -1. When balance goes negative, reverse one '}' to '{', count++, balance += 2. At end, balance positive: need to reverse half of remaining '{' to '}'.

```python
def count_reversals(s):
    if len(s) % 2 != 0:
        return -1
    open_count = 0
    close_count = 0
    for c in s:
        if c == '{':
            open_count += 1
        else:
            if open_count > 0:
                open_count -= 1
            else:
                close_count += 1
    return (open_count + 1) // 2 + (close_count + 1) // 2
```
