# Asymptotic Notations

Asymptotic notation describes the growth rate of functions as the input size approaches infinity. It abstracts away constant factors and lower-order terms to focus on dominant behavior.

## Big O (Upper Bound)

**Formal definition**: f(n) = O(g(n)) if there exist constants c > 0 and n0 > 0 such that 0 <= f(n) <= c * g(n) for all n >= n0.

**Intuition**: f grows at most as fast as g (up to a constant factor). Big O provides an upper bound on growth.

- 3n + 5 = O(n): choose c = 4, n0 = 5; then 3n + 5 <= 4n for n >= 5
- 2n^2 + 3n + 1 = O(n^2): choose c = 3, n0 = 4
- log n = O(n): log grows slower than any positive power of n

## Big Omega (Lower Bound)

**Formal definition**: f(n) = Omega(g(n)) if there exist constants c > 0 and n0 > 0 such that 0 <= c * g(n) <= f(n) for all n >= n0.

**Intuition**: f grows at least as fast as g. Big Omega provides a lower bound.

- 3n + 5 = Omega(n): choose c = 3, n0 = 1; then 3n <= 3n + 5 for n >= 1
- n^2 - n = Omega(n^2): for large n, n^2 dominates

## Big Theta (Tight Bound)

**Formal definition**: f(n) = Theta(g(n)) if f(n) = O(g(n)) and f(n) = Omega(g(n)).

**Intuition**: f grows at the same rate as g (within constant factors). Big Theta provides a tight bound.

- 3n + 5 = Theta(n)
- 2n^2 + 3n + 1 = Theta(n^2)
- Merge sort: Theta(n log n) in all cases

## Little o and Little omega

**Little o**: f(n) = o(g(n)) if lim_{n->inf} f(n)/g(n) = 0. f grows strictly slower than g.

- n = o(n^2)
- log n = o(n)
- 1 = o(log n)

**Little omega**: f(n) = omega(g(n)) if lim_{n->inf} f(n)/g(n) = infinity. f grows strictly faster than g.

- n^2 = omega(n)
- 2^n = omega(n^k) for any constant k

## Comparison Rules

- O is like <=, Omega is like >=, Theta is like =
- f = O(g) implies g = Omega(f)
- f = Theta(g) iff g = Theta(f)
- Transitivity: f = O(g) and g = O(h) implies f = O(h)

## Limit-Based Proofs

To determine the relationship between f and g, compute L = lim_{n->inf} f(n)/g(n):

- L = 0: f = o(g), f = O(g)
- 0 < L < infinity: f = Theta(g)
- L = infinity: f = omega(g), f = Omega(g)

**Example**: f(n) = n log n, g(n) = n^2

L = lim (n log n) / n^2 = lim (log n) / n = 0 (L'Hopital: 1/n / 1 = 0)

Thus n log n = o(n^2).

## Dominance Rules

1. **Drop constants**: 5n = O(n), 1000 = O(1)
2. **Drop lower-order terms**: n^2 + n + 1 = O(n^2)
3. **Logarithm base doesn't matter**: log_2 n = Theta(log_10 n) = Theta(ln n)
4. **Dominance hierarchy**: 1 < log n < sqrt(n) < n < n log n < n^2 < n^3 < 2^n < n!

## Common Complexities Hierarchy

O(1) < O(log n) < O(sqrt(n)) < O(n) < O(n log n) < O(n^2) < O(n^3) < O(2^n) < O(n!)

Each function in this list grows strictly slower than the next (in the limit).

## Worked Examples of Classifying Functions

**Example 1**: Classify f(n) = 5n^3 + 2n^2 + 100n + 7

- Dominant term: 5n^3
- Drop constant: n^3
- Answer: Theta(n^3), O(n^3), Omega(n^3)

**Example 2**: Classify f(n) = 2^n + n^10

- 2^n dominates n^10 for large n
- Answer: Theta(2^n)

**Example 3**: Classify f(n) = log(n^2) + log(n^3)

- log(n^2) + log(n^3) = 2 log n + 3 log n = 5 log n
- Answer: Theta(log n)

**Example 4**: Classify f(n) = sqrt(n) * log n

- sqrt(n) = n^(1/2), so sqrt(n) * log n is between n^(1/2) and n
- sqrt(n) * log n = o(n) since log n = o(n^epsilon) for any epsilon > 0
- sqrt(n) * log n = omega(sqrt(n)) since log n -> infinity
- Answer: Theta(sqrt(n) * log n), O(n), Omega(sqrt(n))

**Example 5**: Compare n^1.5 and n log n

- lim (n^1.5) / (n log n) = lim n^0.5 / log n = infinity
- n^1.5 grows faster
- n log n = o(n^1.5)
