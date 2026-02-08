# Combinatorics and Counting

## Problem 1: Stars and Bars

**Statement:** How many ways to distribute $n$ identical objects into $k$ distinct boxes?

**Solution:** $\binom{n+k-1}{k-1}$ ways.

**Key insight:** Use stars (objects) and bars (dividers between boxes).

Example: Distribute 5 objects into 3 boxes: $**|**|*$ means box1 gets 2, box2 gets 2, box3 gets 1.

We need $k-1$ bars among $n$ stars: $\binom{n+k-1}{k-1}$ arrangements.

**Formula:** $\binom{n+k-1}{k-1} = \binom{n+k-1}{n}$.

## Problem 2: Pigeonhole Principle

**Statement:** If $n+1$ pigeons are placed into $n$ holes, at least one hole contains at least 2 pigeons.

**Generalization:** If $kn+1$ objects are placed into $n$ boxes, at least one box contains at least $k+1$ objects.

**Application:** In any group of 367 people, at least two share a birthday (366 possible birthdays + 1).

**Proof:** By contradiction. If each box has at most $k$ objects, total is at most $kn < kn+1$, contradiction.

## Problem 3: Derangements

**Statement:** How many permutations of $\{1,2,\ldots,n\}$ have no fixed points? (Derangements)

**Solution:** $!n = n! \sum_{k=0}^n \frac{(-1)^k}{k!}$.

**Recurrence:** $!n = (n-1)(!(n-1) + !(n-2))$.

**Approximation:** $!n \approx n!/e$ for large $n$.

**Example:** $!4 = 9$ derangements of 4 elements.

**Inclusion-exclusion:**
$$!n = \sum_{k=0}^n (-1)^k \binom{n}{k} (n-k)! = n! \sum_{k=0}^n \frac{(-1)^k}{k!}$$

## Problem 4: Catalan Numbers

**Statement:** How many valid parentheses sequences of length $2n$? (e.g., $n=2$: ()(), (()))

**Solution:** $C_n = \frac{1}{n+1}\binom{2n}{n}$.

**Recurrence:** $C_n = \sum_{i=0}^{n-1} C_i C_{n-1-i}$ with $C_0 = 1$.

**First values:** $C_0=1$, $C_1=1$, $C_2=2$, $C_3=5$, $C_4=14$.

**Applications:**
- Valid parentheses: $C_n$
- Binary trees with $n$ nodes: $C_n$
- Paths not crossing diagonal: $C_n$
- Triangulations of polygon: $C_{n-2}$

## Problem 5: Stirling Numbers

**Stirling numbers of second kind:** $S(n,k)$ = ways to partition $n$ objects into $k$ non-empty subsets.

**Recurrence:** $S(n,k) = k S(n-1,k) + S(n-1,k-1)$.

**Formula:** $S(n,k) = \frac{1}{k!} \sum_{j=0}^k (-1)^{k-j} \binom{k}{j} j^n$.

**Example:** $S(4,2) = 7$ (ways to partition 4 objects into 2 groups).

**Stirling numbers of first kind:** $s(n,k)$ = permutations of $n$ with $k$ cycles.

**Recurrence:** $s(n,k) = (n-1)s(n-1,k) + s(n-1,k-1)$.

## Problem 6: Grid Path Counting

**Statement:** How many paths from $(0,0)$ to $(m,n)$ moving only right and up?

**Solution:** $\binom{m+n}{m} = \binom{m+n}{n}$.

**Key insight:** Need $m$ rights and $n$ ups in any order: $\binom{m+n}{m}$.

**Example:** $(0,0)$ to $(3,4)$: $\binom{7}{3} = 35$ paths.

## Problem 7: Lattice Paths with Obstacles

**Statement:** How many paths from $(0,0)$ to $(m,n)$ avoiding obstacles?

**Solution:** Use inclusion-exclusion or dynamic programming.

**DP approach:** $dp[i][j] = dp[i-1][j] + dp[i][j-1]$ if $(i,j)$ not obstacle, else 0.

**Inclusion-exclusion:** Count all paths, subtract those through obstacles, add back intersections, etc.

**Example:** Grid with obstacle at $(2,2)$:
- Total: $\binom{4}{2} = 6$
- Through obstacle: $\binom{2}{1} \times \binom{2}{1} = 4$
- Valid: $6 - 4 = 2$ (need to check for double counting).

## Problem 8: Subset Counting

**Problem 8a:** How many subsets of $\{1,2,\ldots,n\}$?

**Solution:** $2^n$ (each element: include or not).

**Problem 8b:** How many subsets of size $k$?

**Solution:** $\binom{n}{k}$.

**Problem 8c:** How many subsets with even sum?

**Solution:** For $n \geq 1$: $2^{n-1}$.

**Key insight:** Pair subsets that differ only in element 1. One has even sum, one has odd sum.

## Problem 9: Partition Problems

**Problem 9a:** Integer partitions: ways to write $n$ as sum of positive integers (order doesn't matter).

**Example:** $p(4) = 5$: 4, 3+1, 2+2, 2+1+1, 1+1+1+1.

**Recurrence:** Complex, uses generating functions.

**Problem 9b:** Compositions: ways to write $n$ as sum (order matters).

**Solution:** $2^{n-1}$ compositions of $n$.

**Key insight:** Place $n-1$ plus signs between $n$ ones: $2^{n-1}$ choices (include/exclude each plus).

## Problem 10: Multinomial Coefficients

**Statement:** Ways to arrange $n$ objects where there are $n_1$ of type 1, $n_2$ of type 2, ..., $n_k$ of type $k$?

**Solution:** $\binom{n}{n_1, n_2, \ldots, n_k} = \frac{n!}{n_1! n_2! \cdots n_k!}$.

**Example:** Arrange letters in MISSISSIPPI:
- M:1, I:4, S:4, P:2
- Ways: $\frac{11!}{1! 4! 4! 2!} = 34,650$.

**Relation to binomial:** $\binom{n}{k} = \binom{n}{k, n-k}$.

## Problem 11: Inclusion-Exclusion

**Statement:** Count elements in union of sets.

**Formula:**
$$|A_1 \cup A_2 \cup \cdots \cup A_n| = \sum_i |A_i| - \sum_{i<j} |A_i \cap A_j| + \sum_{i<j<k} |A_i \cap A_j \cap A_k| - \cdots$$

**Example:** Numbers 1-100 divisible by 2 or 3:
- Divisible by 2: $\lfloor 100/2 \rfloor = 50$
- Divisible by 3: $\lfloor 100/3 \rfloor = 33$
- Divisible by 6: $\lfloor 100/6 \rfloor = 16$
- Answer: $50 + 33 - 16 = 67$.

## Problem 12: Burnside's Lemma

**Statement:** Count distinct colorings under group actions.

**Formula:** Number of orbits = $\frac{1}{|G|} \sum_{g \in G} |\text{Fix}(g)|$.

**Example:** Color vertices of square with 2 colors, rotations equivalent:
- Identity: $2^4 = 16$ fixed
- 90° rotation: $2^1 = 2$ fixed (all same color)
- 180° rotation: $2^2 = 4$ fixed
- 270° rotation: $2^1 = 2$ fixed
- Answer: $(16 + 2 + 4 + 2)/4 = 6$ distinct colorings.

## Problem 13: Generating Functions

**Statement:** Use generating functions to count.

**Example:** Ways to make change for \$1 using coins (1, 5, 10, 25 cents)?

**Generating function:** $(1+x+x^2+\cdots)(1+x^5+x^{10}+\cdots)(1+x^{10}+x^{20}+\cdots)(1+x^{25}+x^{50}+\cdots)$

Coefficient of $x^{100}$ gives answer.

**For infinite supply:** $(1-x)^{-1}(1-x^5)^{-1}(1-x^{10})^{-1}(1-x^{25})^{-1}$.

## Problem 14: Recurrence Relations

**Problem 14a:** Fibonacci: $F_n = F_{n-1} + F_{n-2}$, $F_0=0$, $F_1=1$.

**Closed form:** $F_n = \frac{\phi^n - \psi^n}{\sqrt{5}}$ where $\phi = \frac{1+\sqrt{5}}{2}$, $\psi = \frac{1-\sqrt{5}}{2}$.

**Problem 14b:** Towers of Hanoi: $T_n = 2T_{n-1} + 1$, $T_1 = 1$.

**Solution:** $T_n = 2^n - 1$.

## Problem 15: Permutations with Restrictions

**Problem 15a:** Permutations with no adjacent elements same.

**Example:** Arrange 1,1,2,2,3,3 with no adjacent identical:
- Use inclusion-exclusion
- Total: $6!/(2!2!2!) = 90$
- Subtract those with adjacent 1s, 2s, 3s
- Add back intersections
- Result: Complex but computable.

**Problem 15b:** Circular permutations: $n$ objects arranged in circle.

**Solution:** $(n-1)!$ (fix one, arrange rest).

## Problem 16: Graph Counting

**Problem 16a:** Labeled graphs on $n$ vertices.

**Solution:** $2^{\binom{n}{2}}$ (each pair: edge or no edge).

**Problem 16b:** Trees on $n$ labeled vertices.

**Solution:** $n^{n-2}$ (Cayley's formula).

**Problem 16c:** Spanning trees of complete graph $K_n$.

**Solution:** $n^{n-2}$.

## Problem 17: Ball and Urn Problems

**Problem 17a:** Distribute $n$ distinct balls into $k$ distinct urns.

**Solution:** $k^n$ (each ball: $k$ choices).

**Problem 17b:** Distribute $n$ identical balls into $k$ distinct urns.

**Solution:** $\binom{n+k-1}{k-1}$ (stars and bars).

**Problem 17c:** $n$ distinct balls into $k$ identical urns (non-empty).

**Solution:** $S(n,k)$ (Stirling numbers of second kind).

## Problem 18: Necklace Problems

**Statement:** Count distinct necklaces (rotations/reflections equivalent).

**Use Burnside's lemma** or Polya enumeration.

**Example:** $n$ beads, $k$ colors, rotations equivalent:
- Identity: $k^n$ fixed
- Rotation by $i$: $k^{\gcd(n,i)}$ fixed (cycles of length $n/\gcd(n,i)$)
- Answer: $\frac{1}{n} \sum_{i=0}^{n-1} k^{\gcd(n,i)}$.

## Problem 19: Lattice Path Enumeration

**Problem 19a:** Paths staying above diagonal (ballot problem).

**Solution:** $\frac{m-n+1}{m+1}\binom{m+n}{n}$ (if $m \geq n$).

**Problem 19b:** Paths touching diagonal exactly $k$ times.

**Use reflection principle** or generating functions.

## Problem 20: Advanced Counting

**Problem 20a:** Ways to tile $2 \times n$ board with $1 \times 2$ dominoes.

**Recurrence:** $a_n = a_{n-1} + a_{n-2}$ (Fibonacci).

**Problem 20b:** Ways to parenthesize $n+1$ factors.

**Solution:** $C_n$ (Catalan numbers).

**Problem 20c:** Binary trees with $n$ nodes.

**Solution:** $C_n$.

**Problem 20d:** Ways to triangulate $(n+2)$-gon.

**Solution:** $C_n$.

Many problems reduce to Catalan numbers!
