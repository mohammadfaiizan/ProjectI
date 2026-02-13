# Greedy - Definition and Concepts

## Greedy Choice Property

A greedy algorithm makes the choice that looks best at the moment. The key property is:

**Locally optimal choices lead to globally optimal solution.**

At each step, the algorithm selects the option that appears best without reconsidering previous choices. This works only when the problem exhibits the greedy choice property: making the locally optimal choice at each step yields a globally optimal solution.

## Optimal Substructure

A problem has optimal substructure if an optimal solution to the problem contains optimal solutions to its subproblems. Both greedy and dynamic programming require this property. The difference is in how they use it:

- **Greedy**: Makes one irrevocable choice, then solves the remaining subproblem
- **DP**: Considers all choices and picks the best among them

## Greedy vs Dynamic Programming

| Aspect | Greedy | Dynamic Programming |
|--------|--------|---------------------|
| Choices | One choice per step, no reconsideration | All choices considered, optimal among subproblems |
| Subproblems | Solve one reduced subproblem | Solve overlapping subproblems, memoize |
| Proof | Need to prove greedy choice is safe | Optimal substructure + overlapping subproblems |
| Examples | Activity selection, Huffman, Dijkstra | 0/1 knapsack, LCS, coin change (general) |
| When to use | Greedy choice property holds | Need to explore multiple choices |

## Proof Techniques for Greedy Correctness

### Exchange Argument

Show that any optimal solution can be transformed into our greedy solution without worsening it. If we can exchange our greedy choice for the optimal choice and the result is no worse, then greedy is correct.

### Greedy Stays Ahead

Prove that after each step, our greedy solution is at least as good as any optimal solution up to that point. Use induction: base case holds, and if we stay ahead after step k, we stay ahead after step k+1.

### Structural Argument

Prove that there exists an optimal solution that includes our first greedy choice. Then the problem reduces to a smaller instance with the same structure.

## When Greedy Works

- **Activity selection**: Choosing earliest-finishing non-overlapping activity is optimal
- **Fractional knapsack**: Taking items by value/weight ratio is optimal
- **Huffman coding**: Merging two smallest frequencies minimizes expected code length
- **Dijkstra**: Relaxing closest unvisited vertex gives shortest path
- **MST (Kruskal/Prim)**: Adding minimum weight safe edge is optimal

## When Greedy Fails

- **0/1 Knapsack**: Greedy by value/weight fails; need DP
- **Coin change (general)**: Greedy fails for denominations like [1, 3, 4] and target 6 (greedy: 4+1+1=3 coins; optimal: 3+3=2 coins)
- **Job scheduling with deadlines (weighted)**: Simple earliest-deadline fails; need specialized greedy or DP
- **Traveling salesman**: Nearest-neighbor greedy gives suboptimal tours

## Common Greedy Strategies

### Sort First

Many greedy algorithms begin by sorting: by end time (intervals), by value/weight (knapsack), by deadline, by start time. The sort order encodes the greedy criterion.

### Pick Extremes

Choose minimum or maximum of some criterion: earliest finish, largest value/weight ratio, smallest/largest element. Often combined with a heap for efficient repeated min/max.

### Frequency-Based

Use character or element frequency to guide choices: Huffman (merge smallest frequencies), reorganize string (place most frequent first with gaps).

### Sweep Line

Process events in chronological order; at each event, make greedy decision (e.g., assign room, burst balloon).

### Two Pointers

Maintain two indices; advance based on comparison (merge sorted arrays, container with most water).

## Time Complexity Considerations

Greedy algorithms are often efficient because they make one pass or a few passes after sorting:

| Pattern | Typical Complexity |
|---------|-------------------|
| Sort + single pass | O(n log n) |
| Heap-based (k operations) | O(n log k) |
| Sweep line with events | O(n log n) |
| Two pointers | O(n) |
