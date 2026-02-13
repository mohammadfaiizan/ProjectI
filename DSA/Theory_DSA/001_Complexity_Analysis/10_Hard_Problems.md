# Hard Complexity Analysis Problems

## Problem 1

Solve the recurrence T(n) = 2T(n/2) + Theta(n^2) using the Master Theorem. What is T(n)?

**Answer**: T(n) = Theta(n^2). a=2, b=2, log_b a = 1. f(n) = n^2 = Omega(n^{1+epsilon}) for epsilon=1. Regularity: 2*(n/2)^2 = n^2/2 <= c*n^2 for c=1/2 < 1. Case 3 applies. T(n) = Theta(n^2).

---

## Problem 2

Prove that T(n) = 3T(n/2) + Theta(n) yields T(n) = Theta(n^{log_2 3}) using the substitution method.

**Answer**: Guess T(n) <= c * n^{log_2 3} - d*n for constants c, d. Substitute: T(n) = 3T(n/2) + n <= 3(c*(n/2)^{log_2 3} - d*n/2) + n = 3c * n^{log_2 3} / 2^{log_2 3} - 3d*n/2 + n = c * n^{log_2 3} - (3d/2 - 1)*n. We need (3d/2 - 1) >= d, so d >= 2. Choose d=2. Then T(n) <= c * n^{log_2 3} - 2n. For base case, choose c large enough. Similarly for lower bound. Thus T(n) = Theta(n^{log_2 3}).

---

## Problem 3

Analyze the amortized cost of a sequence of n operations on a stack that supports push (O(1)), pop (O(1)), and multipop(k) which pops min(k, stack size) elements. Use the accounting method.

**Answer**: Assign amortized cost 2 to push, 0 to pop and multipop. Each push overpays by 1; that credit stays on the pushed element. When we pop (via pop or multipop), the popped element's credit pays for the actual cost. Credit invariant: credits = stack size. Each push adds 1 to stack and 1 credit. Each pop removes 1 from stack and uses 1 credit. Amortized O(1) per operation.

---

## Problem 4

Consider a dynamic array that triples (instead of doubles) when full. Show that append still has O(1) amortized cost using the aggregate method.

**Answer**: After n appends from empty, let resizes occur at sizes 1, 3, 9, 27, ... up to n. Copy costs: 1 + 3 + 9 + ... + n/3. Sum = (3^k - 1)/(3 - 1) for k such that 3^k = n, so sum = Theta(n). Plus n for the n appends. Total O(n). Amortized O(1) per append.

---

## Problem 5

Prove that 2^(n+1) = O(2^n). Is 2^(2n) = O(2^n)?

**Answer**: 2^(n+1) = 2 * 2^n. Choose c=2, n0=1. Then 2^(n+1) <= 2 * 2^n for all n >= 1. So 2^(n+1) = O(2^n). For 2^(2n) = 2^n * 2^n: lim 2^(2n)/2^n = lim 2^n = infinity. So 2^(2n) is NOT O(2^n). In fact 2^n = o(2^(2n)).

---

## Problem 6

Solve T(n) = T(sqrt(n)) + 1. Hint: substitute n = 2^k.

**Answer**: Let n = 2^k. Then T(2^k) = T(2^(k/2)) + 1. Let S(k) = T(2^k). S(k) = S(k/2) + 1. By Master Theorem or recursion tree: S(k) = Theta(log k). So T(n) = Theta(log log n).

---

## Problem 7

Analyze the amortized cost of a binary counter that supports only increment. Starting from 0, n increments flip some bits each time. What is the total number of bit flips?

**Answer**: Aggregate method: bit 0 flips every increment (n times). Bit 1 flips every 2 increments (n/2 times). Bit j flips every 2^j increments (n/2^j times). Total flips = n + n/2 + n/4 + ... = n * (1 + 1/2 + 1/4 + ...) <= 2n. Amortized O(1) per increment.

---

## Problem 8

Prove that the following recurrence does not fall into any Master Theorem case: T(n) = 2T(n/2) + n log n.

**Answer**: a=2, b=2, log_b a = 1. f(n) = n log n. Compare with n^1: n log n is asymptotically larger than n (log n -> infinity), so not Case 1. n log n is not Theta(n) (log n factor), so not Case 2. For Case 3, we need f(n) = Omega(n^{1+epsilon}). n log n = Omega(n) but for any epsilon>0, n log n = o(n^{1+epsilon}) since log n = o(n^epsilon). So regularity fails: 2*(n/2)*log(n/2) = n log(n/2) which is not <= c * n log n for c<1 (ratio tends to 1). Use recursion tree: each level does n log n work (with decreasing n), or use extended Master Theorem: T(n) = Theta(n log^2 n).

---

## Problem 9

Why is the decision version of the Traveling Salesman Problem (does a tour of length <= k exist?) NP-complete? Explain the reduction idea.

**Answer**: TSP is in NP: given a tour, we can verify its length in polynomial time. TSP is NP-hard: reduce from Hamiltonian Cycle. Given graph G, form complete graph G' with edge weights 1 for edges in G and 2 for edges not in G. G has a Hamiltonian cycle iff G' has a TSP tour of length n. The reduction is polynomial. Thus TSP is NP-complete.

---

## Problem 10

Analyze the time complexity of quicksort in the worst case when the pivot is always the smallest element.

**Answer**: T(n) = T(0) + T(n-1) + Theta(n) = T(n-1) + Theta(n). Unrolling: T(n) = n + (n-1) + ... + 1 = Theta(n^2). The partition does Theta(n) work, and one subproblem has size 0, the other n-1.

---

## Problem 11

What is the space complexity of merge sort? Consider both the recursive implementation and the array allocation.

**Answer**: O(n) for the auxiliary array used during merge. Recursion depth O(log n), each frame O(1). Total space O(n) dominated by the merge buffer. In-place merge sort variants exist but are more complex.

---

## Problem 12

Prove that if f(n) = O(g(n)) and g(n) = O(h(n)), then f(n) = O(h(n)).

**Answer**: f(n) <= c1 * g(n) for n >= n1. g(n) <= c2 * h(n) for n >= n2. For n >= max(n1, n2): f(n) <= c1 * g(n) <= c1 * c2 * h(n). Choose c = c1 * c2, n0 = max(n1, n2). Then f(n) <= c * h(n) for n >= n0. Thus f(n) = O(h(n)).

---

## Problem 13

Consider a hash table with chaining. n insertions are performed. What is the amortized cost per insertion? Assume the table doubles when load factor exceeds 1.

**Answer**: O(1) amortized. Each insertion is O(1) except when resizing. Resize doubles the table and rehashes all elements: O(m) when m elements. Resizes at 1, 2, 4, 8, ... Total rehash cost O(n). Plus n insertions O(n). Total O(n). Amortized O(1) per insertion.

---

## Problem 14

Solve T(n) = T(n - 1) + T(n - 2) + 1 with T(0) = T(1) = 1. Give a tight bound.

**Answer**: This is similar to Fibonacci recurrence. The homogeneous part has characteristic equation r^2 = r + 1, roots (1+sqrt5)/2 and (1-sqrt5)/2. Solution grows as Theta(phi^n) where phi = (1+sqrt5)/2. So T(n) = Theta(phi^n) = Theta(1.618^n).

---

## Problem 15

A data structure supports insert in O(log n) and get_min in O(1). We perform n inserts. What is the total time? If we use a binary heap, get_min is O(1) and insert is O(log n). Is there a better amortized structure?

**Answer**: Total time O(n log n) for n inserts. A binary heap gives O(1) get_min and O(log n) insert. For amortized O(1) insert with O(1) get_min, we could use a Fibonacci heap: insert O(1) amortized, extract-min O(log n) amortized. But get_min (peek without extract) can be O(1) if we maintain a pointer to the minimum. So with a heap, n inserts = O(n log n) total. Amortized per insert remains O(log n) for heap.
