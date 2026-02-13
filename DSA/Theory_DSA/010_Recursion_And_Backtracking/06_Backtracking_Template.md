# Backtracking Template

## General Template (Choose-Explore-Unchoose)

```python
def backtrack(path, choices):
    if is_solution(path):
        record_solution(path)
        return
    for choice in choices:
        if is_valid(choice, path):
            path.append(choice)
            make_choice(choice)
            backtrack(path, get_next_choices(choice, path))
            undo_choice(choice)
            path.pop()
```

**Choose**: Add a candidate to the current solution (path.append, make_choice).
**Explore**: Recurse with the extended solution (backtrack).
**Unchoose**: Remove the candidate to restore state for trying other options (path.pop, undo_choice).

## Decision Tree Visualization (Text)

```
                    []
                   /  \
                 [1]  [2]
                /  \    \
            [1,2] [1,3] [2,3]
              |     |     |
           [1,2,3] ...  ...
```

Each level = one decision. Leaves = complete solutions or pruned branches.

## Pruning Strategies

**Constraint checking**: Before exploring, verify the partial solution satisfies constraints. Skip invalid branches.

**Bound checking**: For optimization (e.g., subset sum), if current partial sum + remaining max cannot beat best, prune.

**Symmetry breaking**: Avoid exploring equivalent solutions. E.g., for subsets, process indices in order to avoid [1,2] and [2,1].

**Ordering heuristics**: Try more promising choices first to find solution faster (does not reduce worst-case complexity).

## Backtracking vs Brute Force

| Aspect | Backtracking | Brute Force |
|--------|--------------|-------------|
| Exploration | Prunes invalid branches | Explores all possibilities |
| Efficiency | Stops early on constraint violation | Completes full enumeration |
| Structure | Decision tree with pruning | Full state space |
| When | Constraint satisfaction | When no pruning possible |

## Time Complexity Analysis

Typically O(branching_factor ^ depth). Each level multiplies by branching factor. Pruning reduces effective branches.

Example: Subsets of n elements. Branching = 2 (include/exclude each). Depth = n. Time = O(2^n).

## Space Complexity

O(depth) for recursion stack. Plus O(path_length) for current solution. Total often O(n) where n is input size.

## Backtracking vs DFS

Backtracking is DFS on an implicit decision tree. DFS explores graph/tree; backtracking builds solution while exploring. Backtracking explicitly undoes choices (backtrack); DFS may not modify shared state.

## Generic Template Function

```python
def backtrack_template(
    path,
    candidates,
    is_complete,
    is_valid,
    make_choice,
    undo_choice,
    get_candidates,
    results
):
    if is_complete(path):
        results.append(path[:])
        return
    for candidate in get_candidates(candidates, path):
        if is_valid(candidate, path):
            make_choice(candidate, path)
            path.append(candidate)
            backtrack_template(
                path,
                candidates,
                is_complete,
                is_valid,
                make_choice,
                undo_choice,
                get_candidates,
                results
            )
            path.pop()
            undo_choice(candidate, path)
```

**Usage example (subsets)**:

```python
def subsets(nums):
    results = []

    def is_complete(path):
        return True

    def is_valid(c, path):
        return True

    def make_choice(c, path):
        pass

    def undo_choice(c, path):
        pass

    def get_candidates(candidates, path):
        start = path[-1] + 1 if path else 0
        return range(start, len(candidates))

    def backtrack(path):
        results.append(path[:])
        for i in get_candidates(nums, path):
            path.append(i)
            backtrack(path)
            path.pop()

    backtrack([])
    return results
```

## Simplified Practical Template

```python
def backtrack(path, start, target):
    if target_met(path, target):
        results.append(path[:])
        return
    for i in range(start, n):
        if constraint_violated(path, i):
            continue
        path.append(choices[i])
        backtrack(path, i + 1, target)
        path.pop()
```

Key: increment start to avoid duplicates (combination), or use used[] for permutations.
