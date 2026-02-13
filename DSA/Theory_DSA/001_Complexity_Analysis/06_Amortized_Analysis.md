# Amortized Analysis

Amortized analysis provides a way to analyze a sequence of operations where some operations may be expensive but occur infrequently. Instead of worst-case per operation, we bound the average cost per operation over the worst-case sequence.

## Why Amortized Differs from Average Case

**Average case** assumes a probability distribution over inputs. The average is over random inputs.

**Amortized** assumes an adversarial sequence of operations. We guarantee that any sequence of n operations takes at most some total cost; amortized cost = total cost / n. No randomness is involved.

Example: Dynamic array append. A single append can cost O(n) when resizing, but over n appends starting from empty, total cost is O(n), so amortized O(1) per append.

## Aggregate Method

Compute total cost of n operations, then divide by n to get amortized cost per operation.

**Dynamic array example**:

Start with capacity 1. On overflow, double capacity and copy all elements. Cost of append: 1 for normal, 1 + k for resize when growing from k to 2k.

Sequence of n appends: Let k be the number of resizes. Resize when size hits 1, 2, 4, 8, ... Total copy cost: 1 + 2 + 4 + ... + n/2 < n. Plus n for the n appends (each does 1 write). Total < 2n. Amortized cost per append: O(1).

## Accounting Method

Assign an amortized cost to each operation. Some operations overpay (credit), others underpay (use credit). Ensure credit never goes negative.

**Stack with multipop example**:

Operations: push (cost 1), pop (cost 1), multipop(k) (cost min(k, s) where s is stack size).

Assign amortized costs: push = 2, pop = 0, multipop = 0.

- Push: actual 1, we charge 2. Extra 1 stored as credit on the element. Each element has 1 credit.
- Pop: actual 1, we charge 0. Use the element's 1 credit to pay.
- Multipop(k): actual min(k,s), we charge 0. Use credits on popped elements (each had 1).

Credit invariant: number of credits = stack size. Push increases stack and adds 1 credit. Pop/multipop decrease stack and use 1 credit per popped element. Invariant maintained. Amortized cost per operation: O(1).

## Potential Method

Define a potential function Phi that maps the data structure state to a non-negative real number. Amortized cost of operation i = actual cost + Phi(D_i) - Phi(D_{i-1}).

Require Phi(D_0) = 0 and Phi(D_n) >= 0 for any sequence. Then sum of actual costs = sum of amortized costs - Phi(D_n) + Phi(D_0) <= sum of amortized costs.

**Dynamic array with potential**:

Let Phi = 2 * (number of elements) - capacity. Initially empty: Phi = 0.

- Normal append (no resize): actual 1, capacity unchanged, elements +1. Delta Phi = 2. Amortized = 1 + 2 = 3.
- Resize append: from k elements, capacity k, to k+1 elements, capacity 2k. Actual = 1 + k (copy). Phi before = 2k - k = k. Phi after = 2(k+1) - 2k = 2 - k. Delta Phi = 2 - k - k = 2 - 2k. Amortized = 1 + k + 2 - 2k = 3 - k. For k >= 1, amortized <= 2.

So amortized O(1) per append.

**Splay tree overview**:

Potential = sum of log(size of subtree) over all nodes. Splaying a node has amortized O(log n) cost. The potential captures "how balanced" the tree is; splaying tends to improve balance, storing credit for future expensive operations.

## Hash Table Rehashing Analysis

When load factor exceeds threshold, rehash: allocate new larger table, reinsert all elements. Cost of rehash: Theta(n).

If we double on each rehash and start with n inserts: rehashes at sizes 1, 2, 4, 8, ... up to n. Total rehash cost: 1 + 2 + 4 + ... + n/2 + n = O(n). Plus n inserts. Total O(n). Amortized O(1) per insert.

## When to Use Amortized Analysis

- Data structures with occasional expensive operations (dynamic array, hash table, splay tree)
- When worst-case per operation is too pessimistic
- When we care about total cost over a sequence
- When operations have a natural "credit" structure (cheap ops build credit for expensive ones)

Amortized analysis does not apply when we need per-operation worst-case guarantees (e.g., real-time systems). For those, use structures with O(log n) worst-case per operation (e.g., balanced trees) rather than amortized O(1) structures.
