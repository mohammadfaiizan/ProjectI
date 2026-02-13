# Easy Recursion and Backtracking Problems

## 1. Power of Two

Check if n is power of 2. n > 0 and (n & (n-1)) == 0, or recurse n/2.

```python
def isPowerOfTwo(n):
    if n <= 0:
        return False
    return (n & (n - 1)) == 0
```

Time: O(1) | Space: O(1)

---

## 2. Power of Three

Check if n is power of 3. Recurse with n/3, base case n==1.

```python
def isPowerOfThree(n):
    if n <= 0:
        return False
    while n % 3 == 0:
        n //= 3
    return n == 1
```

Time: O(log n) | Space: O(1)

---

## 3. Power of Four

Check if n is power of 4. Power of 2 and (n-1) divisible by 3, or recurse n/4.

```python
def isPowerOfFour(n):
    return n > 0 and (n & (n - 1)) == 0 and (n - 1) % 3 == 0
```

Time: O(1) | Space: O(1)

---

## 4. Reverse String

Reverse string in-place. Swap first and last, recurse on middle.

```python
def reverseString(s):
    def rev(l, r):
        if l >= r:
            return
        s[l], s[r] = s[r], s[l]
        rev(l + 1, r - 1)
    rev(0, len(s) - 1)
```

Time: O(n) | Space: O(n) stack

---

## 5. Swap Nodes in Pairs

Swap every two adjacent nodes in linked list. Recurse on head.next.next, swap first two.

```python
def swapPairs(head):
    if not head or not head.next:
        return head
    nxt = head.next
    head.next = swapPairs(nxt.next)
    nxt.next = head
    return nxt
```

Time: O(n) | Space: O(n)

---

## 6. Merge Two Sorted Lists

Merge two sorted linked lists. Compare heads, recurse on smaller's next.

```python
def mergeTwoLists(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    if l1.val <= l2.val:
        l1.next = mergeTwoLists(l1.next, l2)
        return l1
    l2.next = mergeTwoLists(l1, l2.next)
    return l2
```

Time: O(n + m) | Space: O(n + m)

---

## 7. Maximum Depth of Binary Tree

Return max depth. 1 + max(left_depth, right_depth).

```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

Time: O(n) | Space: O(h)

---

## 8. Invert Binary Tree

Swap left and right of every node. Swap children, recurse on both.

```python
def invertTree(root):
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

Time: O(n) | Space: O(h)

---

## 9. Same Tree

Check if two trees are identical. Compare roots, recurse on left and right.

```python
def isSameTree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
```

Time: O(n) | Space: O(h)

---

## 10. Symmetric Tree

Check if tree is mirror of itself. Helper comparing left subtree with right subtree.

```python
def isSymmetric(root):
    def mirror(l, r):
        if not l and not r:
            return True
        if not l or not r or l.val != r.val:
            return False
        return mirror(l.left, r.right) and mirror(l.right, r.left)
    return mirror(root.left, root.right) if root else True
```

Time: O(n) | Space: O(h)

---

## 11. Path Sum

Check if root-to-leaf path sums to target. Subtract node value, recurse when both children null check remainder.

```python
def hasPathSum(root, targetSum):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == targetSum
    rem = targetSum - root.val
    return hasPathSum(root.left, rem) or hasPathSum(root.right, rem)
```

Time: O(n) | Space: O(h)

---

## 12. Minimum Depth of Binary Tree

Min depth to leaf. 1 + min(left, right), handle single child.

```python
def minDepth(root):
    if not root:
        return 0
    if not root.left:
        return 1 + minDepth(root.right)
    if not root.right:
        return 1 + minDepth(root.left)
    return 1 + min(minDepth(root.left), minDepth(root.right))
```

Time: O(n) | Space: O(h)

---

## 13. Balanced Binary Tree

Check if height diff of subtrees <= 1. Return (height, balanced) from recursion.

```python
def isBalanced(root):
    def check(node):
        if not node:
            return 0, True
        lh, lb = check(node.left)
        rh, rb = check(node.right)
        h = 1 + max(lh, rh)
        ok = lb and rb and abs(lh - rh) <= 1
        return h, ok
    return check(root)[1]
```

Time: O(n) | Space: O(h)

---

## 14. Convert Sorted Array to BST

Build balanced BST from sorted array. Mid as root, recurse left and right halves.

```python
def sortedArrayToBST(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid+1:])
    return root
```

Time: O(n) | Space: O(log n)

---

## 15. Climbing Stairs

Ways to climb n steps (1 or 2). fib(n) = fib(n-1) + fib(n-2), memoize.

```python
def climbStairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b
```

Time: O(n) | Space: O(1)

---

## 16. Fibonacci Number

Return F(n). Base F(0)=0, F(1)=1, recurse or memoize.

```python
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

Time: O(2^n) | Space: O(n)

---

## 17. Pascal's Triangle

Generate triangle. row[i] = prev[i-1] + prev[i], build row by row.

```python
def generate(numRows):
    res = [[1]]
    for _ in range(numRows - 1):
        prev = res[-1]
        row = [1] + [prev[i] + prev[i+1] for i in range(len(prev)-1)] + [1]
        res.append(row)
    return res
```

Time: O(n^2) | Space: O(1)

---

## 18. Pascal's Triangle II

Get kth row. Build row from previous, O(k) space.

```python
def getRow(rowIndex):
    row = [1]
    for _ in range(rowIndex):
        row = [1] + [row[i] + row[i+1] for i in range(len(row)-1)] + [1]
    return row
```

Time: O(k^2) | Space: O(k)

---

## 19. Subsets

All subsets of array. Backtrack include/exclude each element.

```python
def subsets(nums):
    res = []
    def bt(i, path):
        res.append(path[:])
        for j in range(i, len(nums)):
            path.append(nums[j])
            bt(j + 1, path)
            path.pop()
    bt(0, [])
    return res
```

Time: O(2^n) | Space: O(n)

---

## 20. Letter Case Permutation

Toggle letter case, get all strings. If letter, recurse with upper and lower; if digit, recurse once.

```python
def letterCasePermutation(s):
    res = []
    def bt(i, path):
        if i == len(s):
            res.append(''.join(path))
            return
        if s[i].isalpha():
            for c in [s[i].lower(), s[i].upper()]:
                path.append(c)
                bt(i + 1, path)
                path.pop()
        else:
            path.append(s[i])
            bt(i + 1, path)
            path.pop()
    bt(0, [])
    return res
```

Time: O(2^k * n) | Space: O(n)

---

## 21. Binary Watch

Valid times with n LEDs on. Enumerate hour (0-11) and minute (0-59), count bits.

```python
def readBinaryWatch(turnedOn):
    return [f"{h}:{m:02d}" for h in range(12) for m in range(60)
            if bin(h).count('1') + bin(m).count('1') == turnedOn]
```

Time: O(1) | Space: O(1)

---

## 22. Generate Parentheses

All valid n pairs. Backtrack with open < n and close < open.

```python
def generateParenthesis(n):
    res = []
    def bt(s, o, c):
        if len(s) == 2 * n:
            res.append(s)
            return
        if o < n:
            bt(s + '(', o + 1, c)
        if c < o:
            bt(s + ')', o, c + 1)
    bt('', 0, 0)
    return res
```

Time: O(4^n / sqrt(n)) | Space: O(n)

---

## 23. Combination Sum

Combinations that sum to target (reuse allowed). Backtrack with same index allowed.

```python
def combinationSum(candidates, target):
    res = []
    def bt(i, path, rem):
        if rem == 0:
            res.append(path[:])
            return
        if rem < 0 or i >= len(candidates):
            return
        path.append(candidates[i])
        bt(i, path, rem - candidates[i])
        path.pop()
        bt(i + 1, path, rem)
    bt(0, [], target)
    return res
```

Time: O(2^target) | Space: O(target)

---

## 24. Print 1 to N

Print numbers 1 to n recursively. Recurse n-1 first then print n (head recursion).

```python
def print1ToN(n):
    if n <= 0:
        return
    print1ToN(n - 1)
    print(n)
```

Time: O(n) | Space: O(n)

---

## 25. Print N to 1

Print numbers n to 1 recursively. Print n then recurse n-1 (tail recursion).

```python
def printNTo1(n):
    if n <= 0:
        return
    print(n)
    printNTo1(n - 1)
```

Time: O(n) | Space: O(n)

---

## 26. Sum of Natural Numbers

Sum 1 to n. n + sum(n-1), base n<=0 return 0.

```python
def sumNatural(n):
    if n <= 0:
        return 0
    return n + sumNatural(n - 1)
```

Time: O(n) | Space: O(n)

---

## 27. Factorial

n! recursively. n * factorial(n-1), base n<=1 return 1.

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

Time: O(n) | Space: O(n)

---

## 28. Reverse Linked List

Reverse singly linked list. Recurse to end, reverse pointer on return.

```python
def reverseList(head):
    if not head or not head.next:
        return head
    new_head = reverseList(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

Time: O(n) | Space: O(n)

---

## 29. Search in BST

Find value in BST. Compare with root, recurse left or right.

```python
def searchBST(root, val):
    if not root or root.val == val:
        return root
    return searchBST(root.left, val) if val < root.val else searchBST(root.right, val)
```

Time: O(h) | Space: O(h)

---

## 30. Insert into BST

Insert value. Recurse to null position, create node.

```python
def insertIntoBST(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insertIntoBST(root.left, val)
    else:
        root.right = insertIntoBST(root.right, val)
    return root
```

Time: O(h) | Space: O(h)
