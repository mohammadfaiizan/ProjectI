# Recursion and Backtracking - Definition and Fundamentals

## Recursion

Recursion is a programming technique where a function calls itself to solve a problem by breaking it into smaller subproblems of the same type. The solution to the original problem is built from solutions to these smaller instances.

### Core Components

**Function calls itself**: The defining characteristic. The recursive function invokes itself with modified arguments, typically reducing the problem size.

**Base case (termination condition)**: The condition that stops recursion. Without a base case, recursion continues indefinitely. The base case returns a concrete value without making further recursive calls.

**Recursive case (reduction)**: The part where the function calls itself with a smaller or simpler input. The recursive case must eventually reach the base case.

### Call Stack and Stack Frames

When a function calls itself, each invocation creates a new stack frame. A stack frame contains:
- Local variables
- Parameters
- Return address
- Saved registers

The call stack grows with each recursive call and shrinks when calls return. The maximum depth equals the recursion depth.

### Stack Overflow Risk

If recursion depth exceeds the stack limit (typically 1000-10000 frames depending on language and system), a stack overflow occurs. Causes:
- Missing or incorrect base case
- Base case never reached (infinite recursion)
- Problem size too large for recursive approach

Mitigation: Use iteration for very deep recursion, or increase stack size (not recommended), or convert to tail recursion where the language optimizes it.

### Recursion vs Iteration Comparison

| Aspect | Recursion | Iteration |
|--------|-----------|-----------|
| Code clarity | Often more elegant for tree/graph problems | Can be more verbose |
| Stack usage | Uses call stack (implicit) | Uses heap or no extra space |
| Space complexity | O(depth) for stack | O(1) for simple loops |
| Overhead | Function call overhead per level | Loop overhead only |
| Termination | Base case | Loop condition |
| Natural fit | Divide and conquer, tree traversal | Linear processing, simple loops |
| Debugging | Harder (multiple stack frames) | Easier (single execution path |

When to prefer recursion: Tree/graph traversal, divide and conquer, problems with natural recursive structure (merge sort, quicksort, tree operations).

When to prefer iteration: Simple loops, when stack depth could be large, performance-critical code.

---

## Backtracking

Backtracking is a systematic trial-and-error algorithm for finding solutions to constraint satisfaction problems. It builds a solution incrementally, one piece at a time, and abandons a partial solution (backtracks) as soon as it determines the solution cannot be completed to satisfy the constraints.

### Core Concepts

**Systematic trial-and-error**: Explores the solution space in an organized manner, not randomly. Typically uses depth-first exploration.

**Build solution incrementally**: Adds one choice at a time. Each step extends the current partial solution.

**Abandon when constraint violated**: If adding a choice violates a constraint, that branch is abandoned. The algorithm undoes (backtracks) the last choice and tries the next alternative.

**Decision tree**: The solution space is modeled as a tree. Each node represents a partial solution. Edges represent choices. Leaves are complete solutions or dead ends.

**Pruning**: Cutting off branches that cannot lead to valid solutions. Constraint checking and bound checking enable pruning.

**State space tree**: The complete tree of all possible states. Backtracking explores this tree but prunes invalid branches.

### When to Use Each

**Use recursion when**:
- Problem has natural recursive structure (factorial, fibonacci, tree traversal)
- Subproblems are independent
- No need to undo choices
- Divide and conquer applies

**Use backtracking when**:
- Must find all solutions or one solution satisfying constraints
- Choices affect future choices (constraint propagation)
- Need to explore combinations/permutations with constraints
- Must be able to undo choices (backtrack)
- Examples: N-Queens, Sudoku, subset sum, permutation generation

Backtracking is implemented using recursion. The recursive call explores one branch; returning from the call effectively backtracks to try another branch.
