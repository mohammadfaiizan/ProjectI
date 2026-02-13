# Easy Recursion and Backtracking Problems

## Problem List with Description and Approach Hint

1. **Power of Two** - Check if n is power of 2. Approach: n > 0 and (n & (n-1)) == 0, or recurse n/2.

2. **Power of Three** - Check if n is power of 3. Approach: Recurse with n/3, base case n==1.

3. **Power of Four** - Check if n is power of 4. Approach: Power of 2 and (n-1) divisible by 3, or recurse n/4.

4. **Reverse String** - Reverse string in-place. Approach: Swap first and last, recurse on middle.

5. **Swap Nodes in Pairs** - Swap every two adjacent nodes in linked list. Approach: Recurse on head.next.next, swap first two.

6. **Merge Two Sorted Lists** - Merge two sorted linked lists. Approach: Compare heads, recurse on smaller's next.

7. **Maximum Depth of Binary Tree** - Return max depth. Approach: 1 + max(left_depth, right_depth).

8. **Invert Binary Tree** - Swap left and right of every node. Approach: Swap children, recurse on both.

9. **Same Tree** - Check if two trees are identical. Approach: Compare roots, recurse on left and right.

10. **Symmetric Tree** - Check if tree is mirror of itself. Approach: Helper comparing left subtree with right subtree.

11. **Path Sum** - Check if root-to-leaf path sums to target. Approach: Subtract node value, recurse when both children null check remainder.

12. **Minimum Depth of Binary Tree** - Min depth to leaf. Approach: 1 + min(left, right), handle single child.

13. **Balanced Binary Tree** - Check if height diff of subtrees <= 1. Approach: Return (height, balanced) from recursion.

14. **Convert Sorted Array to BST** - Build balanced BST from sorted array. Approach: Mid as root, recurse left and right halves.

15. **Climbing Stairs** - Ways to climb n steps (1 or 2). Approach: fib(n) = fib(n-1) + fib(n-2), memoize.

16. **Fibonacci Number** - Return F(n). Approach: Base F(0)=0, F(1)=1, recurse or memoize.

17. **Pascal's Triangle** - Generate triangle. Approach: row[i] = prev[i-1] + prev[i], build row by row.

18. **Pascal's Triangle II** - Get kth row. Approach: Build row from previous, O(k) space.

19. **Subsets** - All subsets of array. Approach: Backtrack include/exclude each element.

20. **Letter Case Permutation** - Toggle letter case, get all strings. Approach: If letter, recurse with upper and lower; if digit, recurse once.

21. **Binary Watch** - Valid times with n LEDs on. Approach: Enumerate hour (0-11) and minute (0-59), count bits.

22. **Generate Parentheses** - All valid n pairs. Approach: Backtrack with open < n and close < open.

23. **Combination Sum** - Combinations that sum to target (reuse allowed). Approach: Backtrack with same index allowed.

24. **Print 1 to N** - Print numbers 1 to n recursively. Approach: Recurse n-1 first then print n (head recursion).

25. **Print N to 1** - Print numbers n to 1 recursively. Approach: Print n then recurse n-1 (tail recursion).

26. **Sum of Natural Numbers** - Sum 1 to n. Approach: n + sum(n-1), base n<=0 return 0.

27. **Factorial** - n! recursively. Approach: n * factorial(n-1), base n<=1 return 1.

28. **Reverse Linked List** - Reverse singly linked list. Approach: Recurse to end, reverse pointer on return.

29. **Search in BST** - Find value in BST. Approach: Compare with root, recurse left or right.

30. **Insert into BST** - Insert value. Approach: Recurse to null position, create node.
